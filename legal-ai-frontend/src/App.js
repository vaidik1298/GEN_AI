import React, { useState } from "react";
import {
  Container, Typography, Box, Button, LinearProgress, Alert, Paper
} from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import DownloadIcon from "@mui/icons-material/Download";
import GavelIcon from "@mui/icons-material/Gavel";
import OutputAccordion from "./OutputAccordion";
import axios from "axios";
import "./App.css";

function App() {
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");

  const handleFileChange = async (e) => {
    setError("");
    setOutput(null);
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post("http://localhost:5000/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 600000,
      });
      setOutput(res.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    }
    setLoading(false);
  };

  const formatOutputAsText = (output) => {
  if (!Array.isArray(output)) return "";
  return output
    .map(
      (item, idx) =>
        `Clause ${idx + 1} [${item.category?.toUpperCase() || "UNKNOWN"}]\n` +
        `Original: ${item.original}\n` +
        `Simplified: ${item.simplified}\n` +
        (item.justification ? `Justification: ${item.justification}\n` : "") +
        (item.glossary_matches && item.glossary_matches.length > 0
          ? `Glossary Matches:\n${item.glossary_matches
              .map((g) => `  - ${g.term}: ${g.definition}`)
              .join("\n")}\n`
          : "") +
        "-".repeat(40)
    )
    .join("\n\n");
};

const handleDownload = () => {
  const text = formatOutputAsText(output);
  // Wrap in minimal HTML for Word compatibility
  const html = `<html><body><pre style="font-family:Arial">${text}</pre></body></html>`;
  const blob = new Blob([html], { type: "application/msword" });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", "legal_output.doc");
  document.body.appendChild(link);
  link.click();
  link.remove();
};
  return (
    <div className="App">
      <div className="floating-element-1"></div>
      <div className="floating-element-2"></div>
      <div className="floating-element-3"></div>
      <Container maxWidth="md" className="main-container">
        <Paper elevation={4} className="hero-section">
          <div className="legal-icon">
            <GavelIcon />
          </div>
          <Typography variant="h3" align="center" gutterBottom className="hero-title">
            Legal Document Simplifier
          </Typography>
          <Typography variant="subtitle1" align="center" gutterBottom className="hero-subtitle">
            Transform Complex Legal Documents
          </Typography>
          <Typography variant="body1" align="center" gutterBottom className="hero-description">
            Upload your legal document (PDF, CSV, or JSON) and get a plain English summary with simplified explanations, powered by advanced AI technology.
          </Typography>
          <Box sx={{ textAlign: "center", my: 3 }}>
            <input
              type="file"
              accept=".pdf,.csv,.json"
              id="file-upload"
              style={{ display: "none" }}
              onChange={handleFileChange}
              disabled={loading}
            />
            <label htmlFor="file-upload">
              <Button
                variant="contained"
                color="primary"
                component="span"
                startIcon={<CloudUploadIcon />}
                disabled={loading}
                size="large"
                className="upload-button"
              >
                {loading ? "Processing..." : "Upload Document"}
              </Button>
            </label>
            {fileName && (
              <div className="file-name">
                Selected: <b>{fileName}</b>
              </div>
            )}
          </Box>
          {loading && <LinearProgress className="progress-bar" />}
          {error && <Alert severity="error" className="error-alert">{error}</Alert>}
        </Paper>
        {output && Array.isArray(output) && (
          <Paper elevation={3} className="results-section">
            <Typography variant="h5" gutterBottom className="results-title">
              Processed Clauses
            </Typography>
            <OutputAccordion output={output} />
            <Box sx={{ textAlign: "right", mt: 2 }}>
              <Button
                variant="outlined"
                color="secondary"
                startIcon={<DownloadIcon />}
                onClick={handleDownload}
                className="download-button"
              >
                Download Simplified Document
              </Button>
            </Box>
          </Paper>
        )}
      </Container>
    </div>
  );
}

export default App;