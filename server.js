
const express = require('express');
const path = require('path');
const app = express();

// Cloud Run requires binding to the PORT environment variable
const PORT = process.env.PORT || 8080;

app.use(express.json());

// In a real build environment, 'dist' contains the optimized assets
const buildPath = path.join(__dirname, 'dist');
app.use(express.static(buildPath));
app.use(express.static(__dirname));

// Health check for platform monitors
app.get('/api/health', (req, res) => {
  res.json({ status: 'operational', timestamp: new Date().toISOString() });
});

// Single Page Application Fallback
app.get('*', (req, res) => {
  const indexPath = path.join(buildPath, 'index.html');
  res.sendFile(indexPath, (err) => {
    if (err) {
      // If 'dist' doesn't exist yet, fallback to root index.html
      res.sendFile(path.join(__dirname, 'index.html'));
    }
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`
  --------------------------------------------------
  TruthGuard Production Engine
  Status: Operational
  Port: ${PORT}
  Environment: ${process.env.NODE_ENV || 'cloud-run'}
  --------------------------------------------------
  `);
});
