import React from "react";
import ReactDOM from "react-dom/client";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import SigmaNetwork from "./SigmaNetwork";

const Wrapped = withStreamlitConnection(SigmaNetwork);
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<Wrapped />);
