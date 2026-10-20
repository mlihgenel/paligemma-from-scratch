import React from "react";
import ReactDOM from "react-dom/client";
import "katex/dist/katex.min.css";
import "mafs/core.css";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";
import "./styles/prose.css";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
