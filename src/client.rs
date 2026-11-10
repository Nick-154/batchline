//! A deliberately small HTTP/1.1 client for the load generator and tests.
//!
//! It keeps one connection alive and supports exactly what batchline's own
//! responses need: a status line, headers, and a `Content-Length` body.

use std::io;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufStream};
use tokio::net::TcpStream;

pub struct Connection {
    stream: BufStream<TcpStream>,
    host: String,
}

#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    /// Header value by case-insensitive name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn f32_body(&self) -> Vec<f32> {
        self.body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }
}

fn bad(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

impl Connection {
    pub async fn connect(addr: &str) -> io::Result<Self> {
        let stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;
        Ok(Connection {
            stream: BufStream::new(stream),
            host: addr.to_string(),
        })
    }

    pub async fn get(&mut self, path: &str) -> io::Result<Response> {
        self.request("GET", path, None, &[]).await
    }

    /// POST raw little-endian f32 values.
    pub async fn post_f32(&mut self, path: &str, values: &[f32]) -> io::Result<Response> {
        let body: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.request("POST", path, Some("application/octet-stream"), &body)
            .await
    }

    pub async fn post_json(&mut self, path: &str, json: &str) -> io::Result<Response> {
        self.request("POST", path, Some("application/json"), json.as_bytes())
            .await
    }

    pub async fn request(
        &mut self,
        method: &str,
        path: &str,
        content_type: Option<&str>,
        body: &[u8],
    ) -> io::Result<Response> {
        let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\n", self.host);
        if let Some(ct) = content_type {
            head.push_str(&format!("Content-Type: {ct}\r\n"));
        }
        head.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        self.stream.write_all(head.as_bytes()).await?;
        self.stream.write_all(body).await?;
        self.stream.flush().await?;

        let mut line = String::new();
        self.stream.read_line(&mut line).await?;
        // "HTTP/1.1 200 OK"
        let status = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| bad("malformed status line"))?;

        let mut headers = Vec::new();
        let mut length = 0usize;
        loop {
            line.clear();
            if self.stream.read_line(&mut line).await? == 0 {
                return Err(bad("connection closed in headers"));
            }
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                break;
            }
            let (name, value) = trimmed.split_once(':').ok_or_else(|| bad("malformed header"))?;
            let value = value.trim();
            if name.eq_ignore_ascii_case("content-length") {
                length = value.parse().map_err(|_| bad("bad content-length"))?;
            }
            if name.eq_ignore_ascii_case("transfer-encoding") {
                return Err(bad("chunked responses are not supported"));
            }
            headers.push((name.to_string(), value.to_string()));
        }

        let mut body = vec![0u8; length];
        self.stream.read_exact(&mut body).await?;
        Ok(Response {
            status,
            headers,
            body,
        })
    }
}
