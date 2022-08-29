import React, {useCallback, useEffect, useRef, useState} from 'react';
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
  const [textureNeededPaths, setTextureNeededPaths] = useState<string[]>([])
  const [textureNamePrefix, setTextureNamePrefix] = useState("")
  const textureFile = useRef<HTMLInputElement>(null)
  const onLoadTextureClick = () => {
    textureFile.current?.click()
  }
  const motionFile = useRef<HTMLInputElement>(null)
  const onLoadMotionClick = () => {
    motionFile.current?.click();
  }
  const requestRef = useRef<number>();
  const [playing, setPlaying] = useState(false)
  const playUpdate: FrameRequestCallback = useCallback(async time => {
    if (playing) {
      const client = await clientPromise.current
      client?.redraw()
    }
    requestRef.current = requestAnimationFrame(playUpdate);
  }, [playing])
  const onPlayClick = async () => {
    setPlaying(true)
    const client = await clientPromise.current
    client?.play()
  }
  const onRedrawClick = async () => {
    const client = await clientPromise.current
    client?.update()
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
            const texture_paths = client?.get_texture_names()
            if (texture_paths) {
              setTextureNeededPaths([...textureNeededPaths, ...texture_paths])
            }
          })
        }
      }
    }
  }, [textureNeededPaths, modelFile])
  useEffect(() => {
    if (textureFile.current) {
      textureFile.current.onchange = async () => {
        const files = textureFile.current?.files
        if (!files || files.length === 0) {
          console.log("No Texture Selected")
        } else {
          const client = await clientPromise.current;
          for (let idx = 0; idx < files.length; idx++) {
            const file = files[idx];
            const bytes = await readFile(file);
            const data = new Uint8Array(bytes)
            client?.load_texture(textureNamePrefix + file.name, data, true);
          }
          client?.redraw()
        }
      }
    }
  }, [textureFile, textureNamePrefix])
  useEffect(() => {
    if (motionFile.current) {
      motionFile.current.onchange = async () => {
        const files = motionFile.current?.files
        if (!files || files.length === 0) {
          console.log("No Files Selected")
        } else {
          const file = files[0]
          readFile(file).then(async bytes => {
            const data = new Uint8Array(bytes)
            const client = await clientPromise.current;
            client?.load_model_motion(data);
            client?.redraw();
          })
        }
      }
    }
  }, [motionFile])

  useEffect(() => {
    requestRef.current = requestAnimationFrame(playUpdate);
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current)
      }
    };
  }, [playing, playUpdate]); // Make sure the effect runs only once
  useEffect(() => {
    import('mdanceio').then(module => {
      if (!clientPromise.current) {
        console.log("Creating WasmClient...")
        clientPromise.current = module.WasmClient.new(graph_display_ref.current!, module.Backend.WebGPU)
      }
    })
  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <canvas className="Main-Canvas" ref={graph_display_ref} width={800} height={600}/>
        <input type='file' id='model-file' ref={modelFile} accept=".pmx" style={{display: 'none'}}/>
        <input type='file' id='texture-file' ref={textureFile} multiple={true} accept=".png,.jpg,.tga,.bmp"
               style={{display: 'none'}}/>
        <input type='file' id='motion-file' ref={motionFile} accept=".vmd" style={{display: 'none'}}/>
        <button className="Load-Model-Button" onClick={onLoadModelClick}> Load Model</button>
        <div>
          <div>Texture NEEDED</div>
          <ul className="Texture-list">{textureNeededPaths.map((path) => <li className="Texture-path"
                                                                             key={path}>{path}</li>)}</ul>
        </div>
        <div><span className="Hint">Texture Prefix</span> <input type="text" value={textureNamePrefix} onChange={(e) =>
          setTextureNamePrefix(e.target.value)
        }/></div>
        <div className="Hint">We use the texture file name to match needed texture. If texture file is behind a directory, add the directory path as prefix when loading. </div>
        <button className="Load-Texture-Button" onClick={onLoadTextureClick}> Load Texture</button>
        <button className="Load-Motion-Button" onClick={onLoadMotionClick}> Load Motion</button>
        <button className="Play-Button" disabled={playing} onClick={onPlayClick}> Play</button>
        <button className="Redraw-Button" onClick={onRedrawClick}> Redraw</button>
      </header>
    </div>
  );
}

export default App;
