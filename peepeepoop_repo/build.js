const fs = require('fs');
const path = require('path');

const distDir = path.join(__dirname, 'dist');
if (!fs.existsSync(distDir)) {
  fs.mkdirSync(distDir);
}

const frontendFiles = [
  'lstm-forecast-example.html',
  'lstm-forecast-frontend.js',
];

for (const file of frontendFiles) {
  fs.copyFileSync(
    path.join(__dirname, file),
    path.join(distDir, file)
  );
}

// Copy lstm-forecast-example.html as index.html so root URL works
fs.copyFileSync(
  path.join(__dirname, 'lstm-forecast-example.html'),
  path.join(distDir, 'index.html')
);

console.log('Build complete: frontend files copied to dist/');
