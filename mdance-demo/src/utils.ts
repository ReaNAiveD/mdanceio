export function readFile(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    let reader = new FileReader()

    reader.addEventListener('loadend', e => {
      if (!e.target || typeof e.target.result === "string" || !e.target.result){
        reject()
      } else {
        resolve(e.target.result)
      }
    })
    reader.addEventListener('error', reject)

    reader.readAsArrayBuffer(file)
  })
}
