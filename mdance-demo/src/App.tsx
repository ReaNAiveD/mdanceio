import React, {useEffect, useRef} from 'react';
import './App.css';
import {WasmClient} from 'mdanceio';
import {readFile} from "./utils";

function App() {
  const graph_display_ref = useRef<HTMLCanvasElement>(null);
  const clientPromise = useRef<Promise<WasmClient>>();
  const modelFile = useRef<HTMLInputElement>(null)
  const onLoadModelClick = () => {
    modelFile.current?.click();
  };
  const textureFile = useRef<HTMLInputElement>(null)
  const onLoadTextureClick = () => {
    textureFile.current?.click()
  }
  const onUpdateTextureClick = async () => {
    const client = await clientPromise.current
    client?.update_bind_texture()
  }
  const onRedrawClick = async () => {
    const client = await clientPromise.current
    client?.redraw()
  }

  useEffect(() => {
    if (modelFile.current) {
      modelFile.current.onchange = async () => {
        const files = modelFile.current?.files
        if (!files || files.length === 0) {
          console.log("No Files Selected")
        } else {
          const file = files[0]
          readFile(file).then(async bytes => {
            const data = new Uint8Array(bytes)
            const client = await clientPromise.current;
            client?.load_model(data);
            client?.redraw();
          })
        }
      }
    }
  }, [modelFile.current])
  useEffect(() => {
    if (textureFile.current) {
      textureFile.current.onchange = async () => {
        const files = textureFile.current?.files
        if (!files || files.length === 0) {
          console.log("No Texture Selected")
        } else {
          for (let idx = 0; idx < files.length; idx++) {
            const file = files[idx];
            const bytes = await readFile(file);
            const data = new Uint8Array(bytes)
            const client = await clientPromise.current;
            client?.load_texture(file.name, data, true);
            }
        }
      }
    }
  }, [textureFile.current])

  useEffect(() => {
    import('mdanceio').then(module => {
      if (!clientPromise.current) {
        console.log("Use Effect")
        clientPromise.current = module.WasmClient.new(graph_display_ref.current!, module.Backend.WebGPU)
      }
    })
  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <canvas className="Main-Canvas" ref={graph_display_ref}/>
        <input type='file' id='model-file' ref={modelFile} accept=".pmx" style={{display: 'none'}}/>
        <input type='file' id='texture-file' ref={textureFile} multiple={true} accept=".png,.jpg,.tga,.bmp" style={{display: 'none'}}/>
        <button className="Load-Model-Button" onClick={onLoadModelClick}> Load Model</button>
        <button className="Load-Texture-Button" onClick={onLoadTextureClick}> Load Texture</button>
        <button className="Update-Texture-Button" onClick={onUpdateTextureClick}> Update Texture</button>
        <button className="Redraw-Button" onClick={onRedrawClick}> Redraw</button>
      </header>
    </div>
  );
}

export default App;
