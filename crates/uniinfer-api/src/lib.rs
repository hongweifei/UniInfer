//! HTTP surface for the text service.

pub mod error;
pub mod routes;
pub mod state;
pub mod utf8;

use std::io;

use tokio::net::TcpListener;

pub use state::{ApiConfig, AppState};

/// Serve until Ctrl-C.
pub async fn serve(config: ApiConfig, addr: std::net::SocketAddr) -> io::Result<()> {
    let state = AppState::new(&config).map_err(|error| io::Error::other(error.to_string()))?;
    let app = routes::router(state);
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(%addr, "uniinfer api listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}
