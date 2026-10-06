const { spawn } = require('child_process');
const readline = require('readline');
const path = require('path');

class ModelPredictor {
  constructor() {
    this.child = null;
    this.ready = false;
    this.startPromise = null;
    this.pending = new Map();
    this.nextId = 1;
    this.modelInfo = null;
  }

  start() {
    if (this.startPromise) return this.startPromise;

    this.startPromise = new Promise((resolve, reject) => {
      const python = process.env.PYTHON_BIN || (process.platform === 'win32' ? 'python' : 'python3');
      const workerPath = path.join(__dirname, 'model_worker.py');
      let settled = false;
      const startupTimeout = setTimeout(() => {
        fail(new Error('Model worker did not become ready within 120 seconds'));
      }, 120000);

      const fail = (error) => {
        if (settled) return;
        settled = true;
        clearTimeout(startupTimeout);
        this.child?.kill();
        reject(error);
      };

      this.child = spawn(python, ['-u', workerPath], {
        env: process.env,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      const output = readline.createInterface({ input: this.child.stdout });
      output.on('line', (line) => {
        let message;
        try {
          message = JSON.parse(line);
        } catch {
          console.error('Model worker emitted invalid output');
          return;
        }

        if (message.ready === true) {
          this.ready = true;
          this.modelInfo = message.model;
          if (!settled) {
            settled = true;
            clearTimeout(startupTimeout);
            resolve();
          }
          return;
        }

        if (message.ready === false) {
          fail(new Error(message.error || 'Model worker failed to load the model'));
          return;
        }

        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error));
        else pending.resolve(message.result);
      });

      const errors = readline.createInterface({ input: this.child.stderr });
      errors.on('line', (line) => console.error(`Model worker: ${line}`));

      this.child.once('error', (error) => fail(error));
      this.child.once('exit', (code, signal) => {
        this.ready = false;
        const error = new Error(`Model worker exited (code ${code}, signal ${signal})`);
        fail(error);
        for (const pending of this.pending.values()) pending.reject(error);
        this.pending.clear();
      });
    });

    return this.startPromise;
  }

  predict(text) {
    if (!this.ready || !this.child || this.child.killed) {
      return Promise.reject(new Error('Model worker is not ready'));
    }

    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.child.stdin.write(`${JSON.stringify({ id, text })}\n`, (error) => {
        if (!error) return;
        this.pending.delete(id);
        reject(error);
      });
    });
  }
}

module.exports = ModelPredictor;