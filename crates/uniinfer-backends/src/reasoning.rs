//! Incremental separation of model reasoning from the visible answer.
//!
//! Local chat templates do not share one reasoning format: Qwen uses
//! `<think>...</think>`, some templates use ` thinking... response`, and channel
//! models use `<|channel>thought...<channel|>`. The splitter accepts all of
//! them, including markers split across decoding chunks.

const OPEN_MARKERS: &[&str] = &[
    "<think>",
    " thinking",
    "<|channel>thought",
    "<|thinking|>",
    "<|begin_of_thought|>",
    "<|begin▁of▁thinking|>",
    "<｜begin▁of▁thinking｜>",
];
const CLOSE_MARKERS: &[&str] = &[
    "</think>",
    "<|end_of_thought|>",
    "<|end▁of▁thinking|>",
    "<｜end▁of▁thinking｜>",
    "<channel|>",
    "<|/thinking|>",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Content,
    Reasoning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningEvent {
    Reasoning(String),
    Content(String),
}

impl ReasoningEvent {
    fn is_empty(&self) -> bool {
        match self {
            Self::Reasoning(text) | Self::Content(text) => text.is_empty(),
        }
    }
}

/// Stateful splitter suitable for token/chunk streaming.
pub struct ReasoningSplitter {
    buffer: String,
    phase: Phase,
    implicit: bool,
    finished: bool,
}

impl ReasoningSplitter {
    /// `implicit` is used when a model emits reasoning without an opening
    /// marker, which is common for Qwen thinking templates. The first closing
    /// marker then starts the visible answer.
    #[must_use]
    pub fn new(implicit: bool) -> Self {
        Self {
            buffer: String::new(),
            phase: if implicit {
                Phase::Reasoning
            } else {
                Phase::Content
            },
            implicit,
            finished: false,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<ReasoningEvent> {
        if self.finished {
            return Vec::new();
        }
        self.buffer.push_str(text);
        self.drain(false)
    }

    pub fn finish(&mut self) -> Vec<ReasoningEvent> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        self.drain(true)
    }

    /// Split a complete response in one call.
    #[must_use]
    pub fn split(text: &str, implicit: bool) -> (String, String) {
        let mut splitter = Self::new(implicit);
        let mut reasoning = String::new();
        let mut content = String::new();
        for event in splitter.push(text).into_iter().chain(splitter.finish()) {
            match event {
                ReasoningEvent::Reasoning(text) => reasoning.push_str(&text),
                ReasoningEvent::Content(text) => content.push_str(&text),
            }
        }
        (reasoning, content)
    }

    fn drain(&mut self, flush: bool) -> Vec<ReasoningEvent> {
        let mut events = Vec::new();
        loop {
            if self.phase == Phase::Reasoning && self.implicit {
                if find_any(&self.buffer, OPEN_MARKERS).is_some() {
                    // The template/model did provide an opening marker; its
                    // reasoning can stream normally from here on.
                    self.implicit = false;
                    continue;
                }
                if let Some((index, length)) = find_any(&self.buffer, CLOSE_MARKERS) {
                    if index > 0 {
                        events.push(ReasoningEvent::Reasoning(self.buffer[..index].to_owned()));
                    }
                    self.buffer.drain(..index + length);
                    self.phase = Phase::Content;
                    self.implicit = false;
                    continue;
                }
                if flush && !self.buffer.is_empty() {
                    // No closing marker arrived: this was a plain answer, not
                    // an unterminated reasoning block. Never hide it.
                    events.push(ReasoningEvent::Content(std::mem::take(&mut self.buffer)));
                }
                break;
            }
            let found = match self.phase {
                Phase::Content => find_any(&self.buffer, OPEN_MARKERS).map(|found| (found, false)),
                Phase::Reasoning => find_any_any(&self.buffer, OPEN_MARKERS, CLOSE_MARKERS),
            };
            let Some(((index, length), closing)) = found else {
                let markers = match self.phase {
                    Phase::Content => OPEN_MARKERS,
                    Phase::Reasoning => CLOSE_MARKERS,
                };
                let hold = if flush {
                    0
                } else {
                    suffix_prefix(&self.buffer, markers)
                };
                let safe = self.buffer.len().saturating_sub(hold);
                if safe > 0 {
                    self.push_phase(&mut events, &self.buffer[..safe]);
                    self.buffer.drain(..safe);
                }
                break;
            };
            if index > 0 {
                self.push_phase(&mut events, &self.buffer[..index]);
                self.buffer.drain(..index);
            }
            self.buffer.drain(..length);
            self.phase = if closing {
                Phase::Content
            } else {
                Phase::Reasoning
            };
        }
        events.retain(|event| !event.is_empty());
        events
    }

    fn push_phase(&self, events: &mut Vec<ReasoningEvent>, text: &str) {
        if text.is_empty() {
            return;
        }
        events.push(match self.phase {
            Phase::Content => ReasoningEvent::Content(text.to_owned()),
            Phase::Reasoning => ReasoningEvent::Reasoning(text.to_owned()),
        });
    }
}

fn find_any(buffer: &str, markers: &[&str]) -> Option<(usize, usize)> {
    markers
        .iter()
        .filter_map(|marker| buffer.find(marker).map(|index| (index, marker.len())))
        .min_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)))
}

fn find_any_any(
    buffer: &str,
    opening: &[&str],
    closing: &[&str],
) -> Option<((usize, usize), bool)> {
    let open = find_any(buffer, opening).map(|found| (found, false));
    let close = find_any(buffer, closing).map(|found| (found, true));
    [open, close]
        .into_iter()
        .flatten()
        .min_by(|a, b| a.0.0.cmp(&b.0.0).then_with(|| b.0.1.cmp(&a.0.1)))
}

fn suffix_prefix(buffer: &str, markers: &[&str]) -> usize {
    let max = markers.iter().map(|marker| marker.len()).max().unwrap_or(0);
    for (index, _) in buffer.char_indices() {
        let suffix = &buffer[index..];
        if suffix.len() > max {
            continue;
        }
        if markers.iter().any(|marker| marker.starts_with(suffix)) {
            return suffix.len();
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_explicit_thinking_chunks() {
        let mut split = ReasoningSplitter::new(false);
        let mut events = Vec::new();
        for chunk in ["<thi", "nk>step one", "\nstep two</thi", "nk>answer"] {
            events.extend(split.push(chunk));
        }
        events.extend(split.finish());
        let (reasoning, content) = collect(events);
        assert_eq!(reasoning, "step one\nstep two");
        assert_eq!(content, "answer");
    }

    #[test]
    fn splits_implicit_reasoning_before_close_marker() {
        let mut split = ReasoningSplitter::new(true);
        let mut events = Vec::new();
        for chunk in ["I should ", "check this.", "</thi", "nk>Final answer"] {
            events.extend(split.push(chunk));
        }
        events.extend(split.finish());
        let (reasoning, content) = collect(events);
        assert_eq!(reasoning, "I should check this.");
        assert_eq!(content, "Final answer");
    }

    #[test]
    fn implicit_mode_switches_to_explicit_markers() {
        let mut split = ReasoningSplitter::new(true);
        let mut events = split.push("<thi");
        events.extend(split.push("nk>reason</thi"));
        events.extend(split.push("nk>answer"));
        events.extend(split.finish());
        let (reasoning, content) = collect(events);
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "answer");
    }

    #[test]
    fn leaves_plain_content_untouched() {
        let mut split = ReasoningSplitter::new(false);
        let mut events = split.push("hello ");
        events.extend(split.push("world"));
        events.extend(split.finish());
        let (reasoning, content) = collect(events);
        assert!(reasoning.is_empty());
        assert_eq!(content, "hello world");
    }

    #[test]
    fn unicode_reasoning_does_not_split_a_character() {
        let mut split = ReasoningSplitter::new(true);
        let mut events = split.push("中文推理");
        events.extend(split.finish());
        let (reasoning, content) = collect(events);
        assert!(reasoning.is_empty());
        assert_eq!(content, "中文推理");
    }

    #[test]
    fn unclosed_implicit_reasoning_falls_back_to_content() {
        let mut split = ReasoningSplitter::new(true);
        let mut events = split.push("plain answer");
        events.extend(split.finish());
        let (reasoning, content) = collect(events);
        assert!(reasoning.is_empty());
        assert_eq!(content, "plain answer");
    }

    fn collect(events: Vec<ReasoningEvent>) -> (String, String) {
        let mut reasoning = String::new();
        let mut content = String::new();
        for event in events {
            match event {
                ReasoningEvent::Reasoning(text) => reasoning.push_str(&text),
                ReasoningEvent::Content(text) => content.push_str(&text),
            }
        }
        (reasoning, content)
    }
}
