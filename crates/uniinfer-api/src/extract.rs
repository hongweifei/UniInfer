use axum::Json;
use axum::extract::{FromRequest, Request};
use serde::de::DeserializeOwned;

use crate::error::ApiError;

/// JSON extractor that returns the project error envelope on rejection.
pub(crate) struct ApiJson<T>(pub T);

impl<T, S> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(request, state)
            .await
            .map_err(|rejection| ApiError::bad_request("invalid_json", rejection.body_text()))?;
        Ok(Self(value))
    }
}
