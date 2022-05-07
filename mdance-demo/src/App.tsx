import React, {useEffect, useRef, useState} from 'react';
import './App.css';
import {WasmClient} from "mdanceio";
import {readFile} from "./utils";

function App() {
  const graph_display_ref = useRef<HTMLCanvasElement>(null);
  const [client, initClient] = useState<WasmClient | null>(null);
  const modelFile = useRef<HTMLInputElement>(null)
  const onLoadModelClick = () => {
    modelFile.current?.click();
  };
  const textureFile = useRef<HTMLInputElement>(null)
  const onLoadTextureClick = () => {
    textureFile.current?.click()
  }
  const onUpdateTextureClick = () => {
    client?.update_bind_texture()
  }
  const onRedrawClick = () => {
    client?.redraw()
  }

  useEffect(() => {
    modelFile.current?.addEventListener('change', () => {
      const files = modelFile.current?.files
      if (!files || files.length === 0) {
        console.log("No Files Selected")
      }
      else {
        const file = files[0]
        readFile(file).then(bytes => {
          const data = new Uint8Array(bytes)
          if (client) {
            client.load_model(data);
            client.redraw();
          }
        })
      }
    })
  })
  useEffect(() => {
    textureFile.current?.addEventListener('change', () => {
      const files = textureFile.current?.files
      if (!files || files.length === 0) {
        console.log("No Texture Selected")
      }
      else {
        for (let idx = 0; idx < files.length; idx++) {
          const file = files[idx];
          readFile(file).then(bytes => {
            const data = new Uint8Array(bytes)
            if (client) {
              client.load_texture(file.name, data)
            }
          })
        }
      }
    })
  })

  useEffect(() => {
    if (!client) {
      import('mdanceio').then(module => {
        module.default(undefined).then(output => {
          console.log(output)
          const promise = module.WasmClient.new(graph_display_ref.current!)
          promise.then(client => {
            initClient(client)
          })
        })
      })
    }
  })

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
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
