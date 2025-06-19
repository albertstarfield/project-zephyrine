// src/components/shuttle-displays/AttitudeIndicator.jsx

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrthographicCamera, Line } from '@react-three/drei';
import * as THREE from 'three';

const toRad = (deg) => deg * (Math.PI / 180);

// --- 3D Scene Component ---
const ADIScene = ({ roll, pitch, cmdRoll, cmdPitch }) => {
  const adiGroupRef = useRef(); // A single group for all rotating elements

  // 1. NEW HIGH-FIDELITY SPHERE TEXTURE WITH FULL GRID
  const adiTexture = useMemo(() => {
    const size = 1024; // High resolution for sharp lines
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    // --- Layer 1: Base Colors ---
    ctx.fillStyle = '#404040'; // Ground
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#FFFFFF'; // Sky
    ctx.fillRect(0, 0, size, size / 2);

    // --- Layer 2: Subtle Full-Globe Grid ---
    ctx.strokeStyle = '#888888'; // Lighter gray for the background grid
    ctx.lineWidth = 1;

    // Latitude lines (vertical grid)
    for (let i = 0; i < 360; i += 10) {
      const x = (i / 360) * size;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, size);
      ctx.stroke();
    }
    // Longitude lines (horizontal grid)
    for (let i = -90; i <= 90; i += 10) {
        const y = size / 2 - (i * 4); // Simple linear mapping for grid
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(size, y);
        ctx.stroke();
    }
    
    // --- Layer 3: Prominent Primary Pitch Ladder (drawn on top of the grid) ---
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 3;
    ctx.font = 'bold 24px "IBM Plex Mono"';
    ctx.fillStyle = '#000000';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Horizon Line with Boxed '0'
    ctx.strokeRect(size / 2 - 30, size / 2 - 20, 60, 40);
    ctx.fillText('0', size / 2, size / 2);

    // Main Pitch Ladder
    for (let i = 3; i <= 90; i += 3) {
      const y = size / 2 - (i * 8); // Scale factor for lines
      const yDown = size / 2 + (i * 8);
      const isMajor = i % 9 === 0;
      const lineWidth = isMajor ? 120 : 60;
      
      // Pitch Up lines (Solid)
      ctx.beginPath();
      ctx.moveTo(size / 2 - lineWidth, y);
      ctx.lineTo(size / 2 + lineWidth, y);
      ctx.stroke();

      // Pitch Down lines (Dashed)
      ctx.beginPath();
      ctx.setLineDash([20, 10]);
      ctx.moveTo(size / 2 - lineWidth, yDown);
      ctx.lineTo(size / 2 + lineWidth, yDown);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Vertical Bank Ticks for the main ladder
      if(isMajor) {
          ctx.beginPath();
          ctx.moveTo(size / 2 - lineWidth, y); ctx.lineTo(size/2 - lineWidth, y-20); ctx.stroke();
          ctx.moveTo(size / 2 + lineWidth, y); ctx.lineTo(size/2 + lineWidth, y-20); ctx.stroke();
          ctx.moveTo(size / 2 - lineWidth, yDown); ctx.lineTo(size/2 - lineWidth, yDown+20); ctx.stroke();
          ctx.moveTo(size / 2 + lineWidth, yDown); ctx.lineTo(size/2 + lineWidth, yDown+20); ctx.stroke();
          // Add pitch numbers
          ctx.fillText(String(i), size / 2 - 150, y);
          ctx.fillText(String(i), size / 2 + 150, y);
          ctx.fillText(String(i), size / 2 - 150, yDown);
          ctx.fillText(String(i), size / 2 + 150, yDown);
      }
    }
    return new THREE.CanvasTexture(canvas);
  }, []);
  
  // Magenta Roll Scale Texture (no changes needed)
  const rollScaleTexture = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 4096;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    ctx.font = 'bold 48px "IBM Plex Mono"';
    ctx.fillStyle = '#FFFFFF';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for(let i=0; i<360; i+=1) {
        const x = (i / 360) * 4096;
        if(i % 10 === 0) {
            ctx.fillRect(x, 0, 4, 40);
            const label = String(i / 10).padStart(2, '0');
            ctx.save();
            ctx.translate(x, 80);
            ctx.rotate(Math.PI / 2);
            ctx.fillText(label, 0, 0);
            ctx.restore();
        } else if (i % 5 === 0) {
            ctx.fillRect(x, 0, 2, 20);
        }
    }
    return new THREE.CanvasTexture(canvas);
  }, []);

  useFrame(() => {
    if (adiGroupRef.current) {
      adiGroupRef.current.rotation.x = toRad(pitch);
      adiGroupRef.current.rotation.z = toRad(roll);
    }
  });

  return (
    <>
      <group ref={adiGroupRef}>
        <mesh>
          <sphereGeometry args={[0.9, 64, 64]} />
          <meshBasicMaterial map={adiTexture} side={THREE.BackSide} />
        </mesh>
        <mesh position={[0, 0, 0.01]}>
            <ringGeometry args={[0.92, 1.0, 128]} />
            <meshBasicMaterial map={rollScaleTexture} color="#FF00FF" transparent={true} side={THREE.DoubleSide}/>
        </mesh>
      </group>

      <group>
        <Line points={[[-0.3, 0, 0.1], [-0.1, 0, 0.1]]} color="#00FF00" lineWidth={4} />
        <Line points={[[0.1, 0, 0.1], [0.3, 0, 0.1]]} color="#00FF00" lineWidth={4} />
        <Line points={[[0, 0.15, 0.1], [0, 0.05, 0.1]]} color="#00FF00" lineWidth={4} />
        <mesh position={[0, 0.1, 0.1]}>
          <ringGeometry args={[0.08, 0.12, 32, 1, 0, Math.PI]} />
          <meshBasicMaterial color="#00FF00" side={THREE.DoubleSide} />
        </mesh>

        <group position={[toRad(cmdRoll) * 3, toRad(cmdPitch) * -3, 0.02]}>
          <Line points={[[-0.25, 0, 0.2], [0.25, 0, 0.2]]} color="#FF00FF" lineWidth={5} />
          <Line points={[[0, -0.8, 0.2], [0, 0.8, 0.2]]} color="#FF00FF" lineWidth={5} />
        </group>
      </group>
    </>
  );
};


// --- Main Component with HTML/CSS Overlays ---
const AttitudeIndicator = (props) => {
  const yawRatePct = Math.max(-50, Math.min(50, (props.yaw_rate || 0) * 10)); 
  const sideslipPct = Math.max(-50, Math.min(50, (props.sideslip_angle || 0) * 20));

  return (
    <div className="adi-full-wrapper">
        <div className="adi-webgl-wrapper">
            <Canvas>
                <OrthographicCamera makeDefault position={[0, 0, 10]} zoom={220} />
                <ADIScene {...props} />
            </Canvas>
        </div>
        
        <div className="adi-fixed-overlays">
            <div className="adi-roll-pointer">▼</div>
            <div className="bezel-border"></div> 
            {[60, 120, 240, 300].map(angle => (
                 <div key={angle} className="bezel-mark" style={{transform: `rotate(${angle}deg)`}}>
                    <div className="bezel-label" style={{'--angle': angle}}>{String(angle/10).padStart(2,'0')}</div>
                 </div>
            ))}
            {[45, 135, 225, 315].map(angle => (
                <div key={angle} className="bezel-diamond" style={{transform: `rotate(${angle}deg)`}}>◆</div>
            ))}
            <div className="adi-scale adi-scale-top">
                <span>5</span><div className="scale-line"></div><span>5</span>
                <div className="adi-rate-indicator" style={{transform: `translateX(${yawRatePct}%)`}}>▼</div>
            </div>
            <div className="adi-scale adi-scale-bottom">
                <span>5</span><div className="scale-line"></div><span>5</span>
                <div className="adi-slip-indicator" style={{transform: `translateX(${sideslipPct}%)`}}>▲</div>
            </div>
        </div>
    </div>
  );
};

export default AttitudeIndicator;