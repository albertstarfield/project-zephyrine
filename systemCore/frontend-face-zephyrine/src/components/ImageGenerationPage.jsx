import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext'; // 1. Import useAuth to get the user ID
import '../styles/components/_imageGeneration.css';

const ImageGenerationPage = () => {
  const { user } = useAuth(); // 2. Get the current user
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);
  
  // 3. Add state to store the generated image URLs
  const [generatedImages, setGeneratedImages] = useState([]);

  const handleGenerate = async () => {
    if (!prompt.trim() || !user) {
      setError('Please enter a prompt and make sure you are logged in.');
      return;
    }
    
    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/images/generations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt,
          userId: user.id,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'An unknown error occurred.');
      }

      const result = await response.json();
      
      // Add the new image to the start of our gallery array
      if (result.data && result.data.length > 0) {
        const newImage = {
            id: `gallery_${Date.now()}`, // Simple unique key for React
            url: result.data[0].url,
            prompt: result.data[0].revised_prompt
        };
        setGeneratedImages(prevImages => [newImage, ...prevImages]);
      }

    } catch (err) {
      console.error('Image generation error:', err);
      setError(err.message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="image-generation-container">
    <img src="/img/ProjectZephy023LogoRenewal.png" alt="Project Zephyrine Logo" className="image-gen-page-logo" />

      <div className="image-generation-header">
        <h1>Image Generation</h1>
        <p>Describe the image you want to create. Be as specific as you can.</p>
        <p>Powered by Flux and Stable Diffusion and Zephy Learning Ability</p>
      </div>

      <div className="prompt-area">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="A futuristic cityscape at sunset, with flying cars and neon signs..."
          className="prompt-input"
          disabled={isGenerating}
        />
        <button 
          onClick={handleGenerate} 
          className="generate-button"
          disabled={isGenerating || !prompt.trim()}
        >
          {isGenerating ? 'Generating...' : 'Generate'}
        </button>
      </div>
      
      {error && <div className="error-message chat-error">{error}</div>}

      <div className="gallery-area">
        <h2>Gallery</h2>
        
        {/* 4. Replace placeholder with the actual gallery grid */}
        <div className="gallery-grid">
          {generatedImages.length > 0 ? (
            generatedImages.map(image => (
              <div key={image.id} className="gallery-item">
                <img src={image.url} alt={image.prompt} />
                <div className="gallery-item-overlay">
                    <p className="gallery-item-prompt">{image.prompt}</p>
                </div>
              </div>
            ))
          ) : (
             !isGenerating && <p className="gallery-placeholder-text">Your generated images will appear here.</p>
          )}
          {isGenerating && <div className="gallery-item loading-shimmer"></div>}
        </div>
      </div>
    </div>
  );
};

export default ImageGenerationPage;