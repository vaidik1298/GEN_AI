import React from "react";
import {
  Accordion, AccordionSummary, AccordionDetails,
  Typography, Chip, Box, Divider
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import "./App.css";

function OutputAccordion({ output }) {
  return (
    <div className="accordion-container">
      {output.map((item, idx) => (
        <Accordion key={idx} className="accordion-item">
          <AccordionSummary expandIcon={<ExpandMoreIcon />} className="accordion-summary">
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Chip label={item.category?.toUpperCase() || "UNKNOWN"} className="category-chip" />
              <Typography variant="subtitle1" className="clause-title">
                Clause {idx + 1}
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails className="accordion-details">
            <div className="content-section">
              <span className="content-label">Original:</span>
              <p className="content-text">{item.original}</p>
            </div>
            <Divider className="content-divider" />
            <div className="content-section">
              <span className="content-label">Simplified:</span>
              <p className="content-text">{item.simplified}</p>
            </div>
            {item.justification && (
              <div className="content-section">
                <span className="content-label">Justification:</span>
                <p className="content-text">{item.justification}</p>
              </div>
            )}
            {item.glossary_matches && item.glossary_matches.length > 0 && (
              <div className="glossary-section">
                <span className="content-label">Glossary Matches:</span>
                <ul className="glossary-list">
                  {item.glossary_matches.map((g, i) => (
                    <li key={i} className="glossary-item">
                      <span className="glossary-term">{g.term}</span>: {g.definition}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </AccordionDetails>
        </Accordion>
      ))}
    </div>
  );
}

export default OutputAccordion;