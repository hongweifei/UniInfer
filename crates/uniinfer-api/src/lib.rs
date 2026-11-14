//! HTTP surface for the text service.

mod auth;
pub mod chat;
pub mod error;
pub mod extract;
pub mod realtime;
pub mod routes;
pub mod sse;
pub mod state;
pub mod utf8;

use std::io;

use axum::Router;
use tokio::net::TcpListener;

pub use state::{ApiConfig, AppState};

/// Serve a prebuilt router on an existing listener.
pub async fn serve_listener(listener: TcpListener, app: Router) -> io::Result<()> {
    axum::serve(listener, app).await
}

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
