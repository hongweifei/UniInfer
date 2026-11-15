//! Cooperative cancellation for GGUF weight reads.

use std::io::{self, Read, Seek, SeekFrom};

use uniinfer_core::CancelToken;

pub(crate) struct CancellableReader<R> {
    inner: R,
    cancel: CancelToken,
}

impl<R> CancellableReader<R> {
    pub(crate) fn new(inner: R, cancel: CancelToken) -> Self {
        Self { inner, cancel }
    }
}

impl<R: Read> Read for CancellableReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.cancel.is_cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "model load cancelled",
            ));
        }
        self.inner.read(buffer)
    }
}

impl<R: Seek> Seek for CancellableReader<R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        if self.cancel.is_cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "model load cancelled",
            ));
        }
        self.inner.seek(position)
    }
}
