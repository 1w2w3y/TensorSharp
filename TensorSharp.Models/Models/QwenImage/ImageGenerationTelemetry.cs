// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Logging;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// The <c>generate_image &lt;model&gt;</c> span and the
    /// <c>tensorsharp.image.generation.duration</c> sample of one picture, generated or edited,
    /// from the moment it is accepted until its PNG is saved or it is refused. The server's image
    /// routes and the CLI both use it, so a dashboard sees their pictures alike. The pipeline's
    /// timings (<see cref="RecordTiming"/>) add a child span and a duration sample per phase.
    ///
    /// <para>
    /// What the request asked for goes on the span once it is read (<see cref="SetRequest"/>),
    /// the prompt only with content capture on; the size and step count the picture was made
    /// with replace the requested ones when it is made (<see cref="SetResult"/>). The first
    /// outcome marked is the one recorded, so the precise mark made where a request stops is not
    /// overwritten by the general one made on its way out. One writer at a time: the request
    /// until its worker starts, then the worker.
    /// </para>
    /// </summary>
    public sealed class ImageGenerationTelemetry : IDisposable
    {
        private const string MediaKindImage = "image";
        private const string ModeGenerate = "generate";
        private const string ModeEdit = "edit";

        private readonly Activity? _activity;
        private readonly long _startTimestamp;
        private readonly string _model;
        private readonly string? _backend;
        private readonly string _mode;
        private string? _outcome;
        private string? _errorType;
        private int _steps;
        private int _disposed;
        // The steps of the denoise phase in progress; they go on its span when it ends.
        private readonly List<ActivityEvent> _stepEvents = new();

        private ImageGenerationTelemetry(string model, string? backend, bool generate)
        {
            _model = string.IsNullOrEmpty(model) ? "unknown" : model;
            _backend = string.IsNullOrEmpty(backend) ? null : backend;
            _mode = generate ? ModeGenerate : ModeEdit;
            _startTimestamp = Stopwatch.GetTimestamp();
            if (!TensorSharpTelemetry.Media.HasListeners())
                return;
            // A child of whatever is current where the request is accepted: the HTTP request on
            // the server, nothing in the CLI, where the picture is the root of its own trace.
            _activity = TensorSharpTelemetry.Media.StartActivity("generate_image " + _model, ActivityKind.Internal);
            if (_activity is not { IsAllDataRequested: true })
                return;
            _activity.SetTag(TelemetryAttributes.MediaKind, MediaKindImage);
            _activity.SetTag(TelemetryAttributes.GenAiRequestModel, _model);
            if (_backend != null) _activity.SetTag(TelemetryAttributes.Backend, _backend);
            _activity.SetTag(TelemetryAttributes.ImageMode, _mode);
            if (TelemetryContext.Protocol is { Length: > 0 } protocol) _activity.SetTag(TelemetryAttributes.Protocol, protocol);
            if (TelemetryContext.RequestId is { Length: > 0 } requestId) _activity.SetTag(TelemetryAttributes.RequestId, requestId);
        }

        /// <summary>
        /// Start the span. <paramref name="model"/> is the file name dashboards group by, never a
        /// path; <paramref name="backend"/> the <c>--backend</c> spelling; <paramref name="generate"/>
        /// true for a picture from words, false for an edit of pictures.
        /// </summary>
        public static ImageGenerationTelemetry Start(string model, string? backend, bool generate) =>
            new(model, backend, generate);

        /// <summary>The span, or null when nobody records it.</summary>
        public Activity? Activity => _activity;

        /// <summary>
        /// What the request asked for. Zero means "the model's default" and is left off; the
        /// seed is always set, because zero is a seed like any other. The prompt is put on the
        /// span only when content capture is on.
        /// </summary>
        public void SetRequest(string? prompt, QwenImageParams parameters, int references)
        {
            ArgumentNullException.ThrowIfNull(parameters);
            if (_activity is not { IsAllDataRequested: true })
                return;
            if (parameters.Width > 0) _activity.SetTag(TelemetryAttributes.ImageWidth, parameters.Width);
            if (parameters.Height > 0) _activity.SetTag(TelemetryAttributes.ImageHeight, parameters.Height);
            if (parameters.Steps > 0) _activity.SetTag(TelemetryAttributes.ImageSteps, parameters.Steps);
            if (parameters.CfgScale > 0) _activity.SetTag(TelemetryAttributes.ImageCfg, (double)parameters.CfgScale);
            _activity.SetTag(TelemetryAttributes.ImageSeed, parameters.Seed);
            if (references > 0) _activity.SetTag(TelemetryAttributes.ImageReferences, references);
            if (prompt != null && TensorSharpTelemetry.CaptureMessageContent)
                _activity.SetTag(TelemetryAttributes.ImagePrompt, prompt);
        }

        /// <summary>
        /// A denoising step of <paramref name="total"/>: the number of steps the picture is made
        /// with, which the pipeline resolves when the request leaves it at zero.
        /// </summary>
        public void RecordStep(int total)
        {
            if (total > 0) Volatile.Write(ref _steps, total);
        }

        /// <summary>
        /// Where the per-phase log lines go (<see cref="LogEventIds.ImageGeneratePhase"/>); null
        /// writes none. Set before the picture starts.
        /// </summary>
        public ILogger? Logger { get; set; }

        /// <summary>
        /// One finished part of the picture, as <see cref="QwenImageParams.OnTiming"/> reports it.
        /// A phase or a stage becomes a child span that ends now and a
        /// <c>tensorsharp.image.phase.duration</c> sample; a phase also writes one log line. A
        /// denoising step becomes a <c>tensorsharp.image.denoise.step.duration</c> sample and an
        /// event on the denoise span, and counts as <see cref="RecordStep"/>.
        /// </summary>
        public void RecordTiming(QwenImageTiming timing)
        {
            if (string.IsNullOrEmpty(timing.Phase) || Volatile.Read(ref _disposed) != 0)
                return;
            double seconds = Math.Max(0, timing.Duration.TotalSeconds);
            if (timing.Step > 0)
            {
                RecordStep(timing.TotalSteps);
                TensorSharpMetrics.ImageDenoiseStepDuration.Record(seconds, MetricTags());
                if (_activity is { IsAllDataRequested: true })
                    _stepEvents.Add(new ActivityEvent("denoise step", DateTimeOffset.UtcNow, new ActivityTagsCollection
                    {
                        { TelemetryAttributes.ImageStep, timing.Step },
                        { TelemetryAttributes.ImageStepDuration, seconds },
                    }));
                return;
            }

            var tags = MetricTags();
            tags.Add(TelemetryAttributes.ImagePhase, timing.Phase);
            if (timing.Stage != null) tags.Add(TelemetryAttributes.ImageStage, timing.Stage);
            TensorSharpMetrics.ImagePhaseDuration.Record(seconds, tags);

            if (_activity != null)
            {
                DateTimeOffset end = DateTimeOffset.UtcNow;
                string name = timing.Stage == null ? timing.Phase : timing.Phase + " " + timing.Stage;
                var child = TensorSharpTelemetry.Media.StartActivity(name, ActivityKind.Internal, _activity.Context,
                    startTime: end - TimeSpan.FromSeconds(seconds));
                if (child != null)
                {
                    if (child.IsAllDataRequested)
                    {
                        child.SetTag(TelemetryAttributes.MediaKind, MediaKindImage);
                        child.SetTag(TelemetryAttributes.ImagePhase, timing.Phase);
                        if (timing.Stage != null) child.SetTag(TelemetryAttributes.ImageStage, timing.Stage);
                        if (timing.Stage == null && timing.Phase == QwenImageTiming.PhaseDenoise)
                            foreach (var step in _stepEvents) child.AddEvent(step);
                    }
                    child.SetEndTime(end.UtcDateTime);
                    child.Stop();
                }
            }
            if (timing.Stage == null)
            {
                if (timing.Phase == QwenImageTiming.PhaseDenoise) _stepEvents.Clear();
                Logger?.LogInformation(LogEventIds.ImageGeneratePhase,
                    "image.phase {Phase} model={Model} seconds={Seconds:F3}", timing.Phase, _model, seconds);
            }
        }

        private System.Diagnostics.TagList MetricTags()
        {
            var tags = new System.Diagnostics.TagList
            {
                { TelemetryAttributes.GenAiRequestModel, _model },
                { TelemetryAttributes.ImageMode, _mode },
            };
            if (_backend != null) tags.Add(TelemetryAttributes.Backend, _backend);
            return tags;
        }

        /// <summary>The picture was made at <paramref name="width"/> x <paramref name="height"/>.</summary>
        public void SetResult(int width, int height)
        {
            if (_activity is not { IsAllDataRequested: true })
                return;
            if (width > 0) _activity.SetTag(TelemetryAttributes.ImageWidth, width);
            if (height > 0) _activity.SetTag(TelemetryAttributes.ImageHeight, height);
            int steps = Volatile.Read(ref _steps);
            if (steps > 0) _activity.SetTag(TelemetryAttributes.ImageSteps, steps);
        }

        /// <summary>The client went away or the picture was stopped. Not an error.</summary>
        public void MarkCancelled() => Mark(TelemetryAttributes.OutcomeCancelled, null, null);

        /// <summary>Refused before the picture was made; <paramref name="errorType"/> is a short stable code.</summary>
        public void MarkRefused(string errorType) => Mark(TelemetryAttributes.OutcomeRefused, errorType, null);

        /// <summary>Failed: the exception's type, never its message. A cancellation counts as <see cref="MarkCancelled"/>.</summary>
        public void MarkFailed(Exception exception)
        {
            ArgumentNullException.ThrowIfNull(exception);
            if (exception is OperationCanceledException)
                MarkCancelled();
            else
                Mark(TelemetryAttributes.OutcomeFailed, exception.GetType().FullName ?? exception.GetType().Name, exception);
        }

        private void Mark(string outcome, string? errorType, Exception? failure)
        {
            if (_outcome != null || Volatile.Read(ref _disposed) != 0)
                return;
            _outcome = outcome;
            _errorType = errorType;
            if (_activity == null)
                return;
            if (failure != null)
                _activity.SetStatus(ActivityStatusCode.Error, failure.GetType().Name);
            if (_activity.IsAllDataRequested)
            {
                _activity.SetTag(TelemetryAttributes.Outcome, outcome);
                if (errorType != null) _activity.SetTag(TelemetryAttributes.ErrorType, errorType);
            }
        }

        /// <summary>Record the duration and stop the span, once.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            string outcome = _outcome ?? TelemetryAttributes.OutcomeCompleted;
            var tags = new System.Diagnostics.TagList
            {
                { TelemetryAttributes.GenAiRequestModel, _model },
                { TelemetryAttributes.ImageMode, _mode },
                { TelemetryAttributes.Outcome, outcome },
            };
            if (_backend != null) tags.Add(TelemetryAttributes.Backend, _backend);
            if (_errorType != null) tags.Add(TelemetryAttributes.ErrorType, _errorType);
            TensorSharpMetrics.ImageGenerationDuration.Record(Stopwatch.GetElapsedTime(_startTimestamp).TotalSeconds, tags);
            if (_activity == null)
                return;
            if (_outcome == null && _activity.IsAllDataRequested)
                _activity.SetTag(TelemetryAttributes.Outcome, outcome);
            _activity.Stop();
        }
    }
}
