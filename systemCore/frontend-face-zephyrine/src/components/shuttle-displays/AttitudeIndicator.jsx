// src/components/shuttle-displays/AttitudeIndicator.jsx
import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrthographicCamera, Line } from '@react-three/drei';
import * as THREE from 'three';

const toRad = (deg) => deg * (Math.PI / 180);

const ADIScene = ({ roll, pitch, cmdRoll, cmdPitch }) => {
  const sphereRef = useRef();

  const adiTexture = useMemo(() => {
    const size = 512; // Higher resolution for crisp lines
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    // --- Draw the realistic texture ---
    ctx.fillStyle = '#606060'; // Ground
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#DCDCDC'; // Sky
    ctx.fillRect(0, 0, size, size / 2);

    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 2;
    ctx.font = '24px "IBM Plex Mono"';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'center';

    // Horizon line
    ctx.beginPath();
    ctx.moveTo(0, size / 2);
    ctx.lineTo(size, size / 2);
    ctx.stroke();

    // Pitch Ladder
    for (let i = -90; i <= 90; i += 10) {
      if (i === 0) continue;
      const y = size / 2 - (i * 4); // Scale factor for lines
      const isMajor = i % 30 === 0;
      const lineWidth = isMajor ? 100 : 60;
      
      ctx.beginPath();
      ctx.moveTo(size / 2 - lineWidth / 2, y);
      ctx.lineTo(size / 2 + lineWidth / 2, y);
      ctx.stroke();

      if (isMajor) {
        ctx.fillText(Math.abs(i), size / 2 - 70, y + 8);
        ctx.fillText(Math.abs(i), size / 2 + 70, y + 8);
      }
    }
    return new THREE.CanvasTexture(canvas);
  }, []);

  useFrame(() => {
    if (sphereRef.current) {
      // The sphere rotates opposite to the vehicle's movement
      sphereRef.current.rotation.x = toRad(pitch);
      sphereRef.current.rotation.z = toRad(roll);
    }
  });

  return (
    <>
      <group ref={sphereRef}>
        <mesh>
          <sphereGeometry args={[1, 64, 64]} />
          <meshBasicMaterial map={adiTexture} side={THREE.BackSide} />
        </mesh>
      </group>

      {/* --- Fixed Symbols (drawn on top) --- */}

      {/* Fixed Aircraft Symbol (Green) */}
      <group>
        <Line points={[[-0.3, 0, 0.1], [-0.1, 0, 0.1]]} color="#00FF00" lineWidth={4} />
        <Line points={[[0.1, 0, 0.1], [0.3, 0, 0.1]]} color="#00FF00" lineWidth={4} />
        <Line points={[[0, 0.1, 0.1], [0, 0.0, 0.1]]} color="#00FF00" lineWidth={4} />
      </group>

      {/* Flight Director (Magenta) - moves based on error */}
      <group position={[toRad(cmdRoll - roll) * -3, toRad(cmdPitch - pitch) * 3, 0]}>
         <Line points={[[-0.15, 0, 0.2], [0.15, 0, 0.2]]} color="#FF00FF" lineWidth={4} />
         <Line points={[[0, -0.15, 0.2], [0, 0.15, 0.2]]} color="#FF00FF" lineWidth={4} />
      </group>
    </>
  );
};

const AttitudeIndicator = (props) => {
  return (
    <div className="adi-webgl-wrapper">
      <Canvas>
        <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={200} />
        <ADIScene {...props} />
      </Canvas>
       <div className="adi-roll-indicator">▲</div>
    </div>
  );
};

export default AttitudeIndicator;