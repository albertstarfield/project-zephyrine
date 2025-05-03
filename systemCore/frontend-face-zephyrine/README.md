# Project Adelaide Zephyrine Charlotte - Web UI

This is a web-based version of the Project Zephyrine interface, built with React and Vite.

## Features

- Modern React-based UI
- System information display
- Chat interface simulation
- Responsive design

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm (v6 or later)

### Installation

1. Clone the repository
2. Navigate to the project directory:
   ```
   cd systemCore/project-zephyrine-web
   ```
3. Install dependencies:
   ```
   npm install
   ```

### Development

To start the development server:

```
npm run dev
```

This will start the application on http://localhost:3000

### Building for Production

To build the application for production:

```
npm run build
```

The built files will be in the `dist` directory.

## Project Structure

- `src/` - Source code
  - `components/` - React components
  - `styles/` - CSS files
  - `utils/` - Utility functions
- `public/` - Static assets

## Notes

This is a web-based version of the original Electron application. It simulates the system information that would normally be provided by the Electron backend. In a production environment, you would need to implement a backend service to provide real system information.
