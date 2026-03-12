const express = require("express");
const path = require("path");
const { spawn } = require("child_process");

const DATA_DIR = path.resolve(__dirname, "../data");
const USER_API_SCRIPT = path.join(__dirname, "../frontend_user_api.py");
const RESEARCH_API_SCRIPT = path.join(__dirname, "../frontend_api.py");

const PYTHON_BIN = process.env.PYTHON_BIN || "python";

const runPythonScript = (scriptPath, args = []) =>
  new Promise((resolve, reject) => {
    const python = spawn(PYTHON_BIN, [scriptPath, ...args], { cwd: path.dirname(scriptPath) });
    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    python.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    python.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(stderr.trim() || `Python exited with ${code}`));
      }
      try {
        const payload = JSON.parse(stdout);
        resolve(payload);
      } catch (err) {
        reject(new Error("Unable to parse Python output"));
      }
    });
  });

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

app.post("/api/login", async (req, res) => {
  const { username, password = "" } = req.body;
  if (!username) {
    return res.status(400).json({ message: "Username required" });
  }
  try {
    const payload = await runPythonScript(USER_API_SCRIPT, [
      "login",
      "--username",
      username,
      "--password",
      password,
    ]);
    return res.json({ wallet: payload.wallet });
  } catch (err) {
    return res.status(400).json({ message: err.message || "Login failed" });
  }
});

app.post("/api/signup", async (req, res) => {
  const { username, password, wallet = 0 } = req.body;
  if (!username || !password) {
    return res.status(400).json({ message: "Username and password required" });
  }
  try {
    const payload = await runPythonScript(USER_API_SCRIPT, [
      "signup",
      "--username",
      username,
      "--password",
      password,
      "--wallet",
      String(wallet),
    ]);
    return res.json({ wallet: payload.wallet });
  } catch (err) {
    return res.status(400).json({ message: err.message || "Signup failed" });
  }
});

app.post("/api/wallet", async (req, res) => {
  const { username, amount } = req.body;
  if (!username || typeof amount === "undefined") {
    return res.status(400).json({ message: "Username and amount required" });
  }
  try {
    const payload = await runPythonScript(USER_API_SCRIPT, [
      "wallet",
      "--username",
      username,
      "--amount",
      String(amount),
    ]);
    return res.json({ wallet: payload.wallet });
  } catch (err) {
    return res.status(400).json({ message: err.message || "Failed to update wallet" });
  }
});

app.post("/api/research", async (req, res) => {
  const { scope = "whole", username = "system", wallet = 0 } = req.body;
  try {
    const payload = await runPythonScript(RESEARCH_API_SCRIPT, [
      "--scope",
      scope,
      "--wallet",
      String(wallet),
      "--username",
      username,
    ]);
    return res.json(payload);
  } catch (err) {
    return res.status(500).json({ message: err.message || "Research failed" });
  }
});

app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Node UI listening on http://localhost:${port}`);
});
