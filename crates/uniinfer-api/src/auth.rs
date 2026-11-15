use axum::extract::{Request, State};
use axum::http::Method;
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::middleware::Next;
use axum::response::Response;
use tower_http::cors::AllowOrigin;

use crate::error::ApiError;
use crate::state::AppState;

pub(crate) async fn authorize(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let path = request.uri().path();
    if request.method() == Method::OPTIONS || path == "/health" || path == "/v1/realtime" {
        return Ok(next.run(request).await);
    }
    if let Some(expected) = &state.token {
        let supplied = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "));
        if supplied != Some(expected.as_str()) {
            return Err(ApiError::unauthorized("missing or invalid bearer token"));
        }
    }
    Ok(next.run(request).await)
}

pub(crate) fn cors() -> tower_http::cors::CorsLayer {
    tower_http::cors::CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| {
            origin.to_str().ok().is_some_and(allowed_browser_origin)
        }))
        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::OPTIONS])
        .allow_headers([CONTENT_TYPE, AUTHORIZATION])
        .max_age(std::time::Duration::from_secs(600))
}

fn allowed_browser_origin(origin: &str) -> bool {
    if origin.starts_with("tauri://") {
        return true;
    }
    let Some(rest) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    let host = rest.split(':').next().unwrap_or_default();
    matches!(
        host,
        "localhost" | "127.0.0.1" | "[::1]" | "tauri.localhost"
    )
}
