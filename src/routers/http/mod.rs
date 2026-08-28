//! HTTP router implementations

use futures_util::{Stream, StreamExt};

pub mod dp_utils;
pub mod logprobs_merge;
pub mod openai_router;
pub mod pd_router;
pub mod pd_types;
mod program_adapter;
pub mod router;
pub mod vllm_pd_router;
pub mod vllm_service_discovery;

/// Keep an RAII guard alive until a response stream completes, errors, or is dropped.
fn guarded_stream<S, T, E, G>(stream: S, guard: G) -> impl Stream<Item = Result<T, E>>
where
    S: Stream<Item = Result<T, E>> + Unpin,
{
    futures_util::stream::unfold((stream, Some(guard)), |(mut stream, guard)| async move {
        match stream.next().await {
            Some(Ok(item)) => Some((Ok(item), (stream, guard))),
            Some(Err(error)) => Some((Err(error), (stream, None))),
            None => None,
        }
    })
}
