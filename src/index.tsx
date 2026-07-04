import { render } from "solid-js/web";
import "./index.css";
import { AppProvider } from "./data/store";
import { App } from "./App";

const root = document.getElementById("root");
if (!root) throw new Error("Root element not found");

render(
  () => (
    <AppProvider>
      <App />
    </AppProvider>
  ),
  root,
);
