import React, {useEffect, useRef, useState} from 'react';
import './App.css';
import {WasmClient} from "mdanceio";
import {readFile} from "./utils";

function App() {
  const graph_display_ref = useRef<HTMLCanvasElement>(null);
  const [client, initClient] = useState<WasmClient | null>(null);
  const inputFile = useRef<HTMLInputElement>(null)
  const onLoadModelClick = () => {
    inputFile.current?.click();
  };

  useEffect(() => {
    inputFile.current?.addEventListener('change', () => {
      const files = inputFile.current?.files
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
        <input type='file' id='file' ref={inputFile} accept=".pmx" style={{display: 'none'}}/>
        <button className="Load-Model-Button" onClick={onLoadModelClick}> Load Model</button>
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
