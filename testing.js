const { question } = require('readline-sync')
const { readFileSync, writeFileSync } = require('fs')

function range(value) {
  return [ ...Array(value).keys() ]
}

/*
  {
    umbral,
    weight,
    output
  }
*/

class Perceptron {
  constructor() {
    this.alpha = 0.1 // factor de aprendizaje. Cuanto mayo es, antes aprende pero tiene menor precision
    // array que nos define las capas y por cada capa contiene un subarray de la siguiente forma:
    // [numero de neuronas de la capa, [ umbrales de la actuacion(u) ], [ pesos(w) ], [salidas de las neuronas(a) ]]
    this.layers = []
    this.ntrain = 0 // numero de entrenamientos ejecutados
    this.saveurl = __dirname + '/perceptron.json' // url donde guardaremos la red neuronal una vez entrenada
    this.trainingdata = __dirname + '/training.json' // url donde tenemos la training data
  }

  init() {
    this.build() // construimos la red neuronal
    this.train() // entrenamos la red neuronal

    const [ { neurons } ] = this.layers

    while (true) { // utilizamos la red neuronal
      range(neurons).forEach(i => {
        this.layers[0].outputs[i] = parseFloat(question('entrada ' + i + ' : '))
      })

      this.percep()
    }
  }

  build() { // funcion que contruye la red neuronal
    console.log('***PERCEPTRON***')
    range(parseFloat(question('numero de capas: '))).forEach(i => {
      let layer = {
        neurons: 0,
        umbrals: [],
        weights: [],
        outputs: []
      }
      layer.neurons = parseFloat(question('numero de neuronas en la capa ' + i + ' : '))

      range(layer.neurons).forEach((umbral, j) => {
        layer.umbrals.push(Math.random()) // Creamos umbrales de actuacion aleatorios
        layer.outputs.push(Math.random()) // Creamos salidas aleatorios
        let wc = []

        if(i != 0)
          range(this.layers[i - 1].neurons).forEach(() => {
            wc.push(Math.random()) // creamos pesos aleatorios
          })

        layer.weights.push(wc)
      })

      this.layers.push(layer) // anadimos la capa a la red neuronal
    })
  }

  percep() { // funcion que calcula la salida de la red neuronal ante una serie de entradas
    console.log('salida de las neuronas capa 0:')
    console.log(this.layers[0].outputs)

    range(this.layers.length - 1).forEach(k => {
      range(this.layers[k + 1].neurons).forEach(i => {
        let suma = 0
        range(this.layers[k].neurons).forEach(j => {
          suma += this.layers[k].outputs[j] * this.layers[k + 1].weights[i][j]
        })
        suma += this.layers[k + 1].umbrals[i]
        suma = (1 / (1 + Math.exp(-suma)))
        this.layers[k + 1].outputs[i] = suma
      })

      console.log('salida de las neuronas capa ' + (k + 1) + ' : ')
      console.log(this.layers[k + 1].outputs)
    })
  }

  train() {
    console.log('-+-Comienza el entrenamiento del layers-+-')
    range(parseFloat(question('numero de veces que quiere repetir el entrenamiento: '))).forEach(() => {
      this.readlines()
    })

    this.save()
  }

  readlines() {
    const { trains } = JSON.parse(readFileSync(this.trainingdata, 'utf8'))
    const [ { neurons } ] = this.layers.slice(-1)

    trains.forEach(({ inputs, outputs }, i) => {
      this.ntrain += 1
      console.log('***entrenamiento numero ' + this.ntrain + ' ***')

      let boolean = false
      let numero = ''
      let se = []

      range(neurons).forEach(() => {
        se.push(0)
      })

      this.layers[0].outputs = inputs

      console.log('outputs', outputs)

      outputs.forEach((output, l) => {
        se[l] = output
      })

      console.log('entrada: ' + JSON.stringify(this.layers[0].outputs) + ' salida esperada: ' + JSON.stringify(se))

      this.percep()
      this.backpropagation(se)
    })
  }

  backpropagation(se) {
    // Calculamos los errores y realizamos las correciones oportunas
    const [ { neurons, umbrals, weights, outputs } ] = this.layers.slice(-1)
    const amountLayers = this.layers.length
    let deltam = []

    // empezamos por la ultima capa de neuronas
    range(neurons).forEach(k => {
      deltam.push(outputs[k] * (1 - outputs[k] * (outputs[k] - se[k])))
      range(this.layers.slice(-2)[0].neurons).forEach(n => {
        this.layers[amountLayers - 1].weights[k][n] -=  this.alpha * deltam[k] * this.layers.slice(-2)[0].outputs[n]
      })
      this.layers[amountLayers - 1].umbrals[k] -= this.alpha * deltam[k]
    })

    // seguimos con las capas ocultas de neuronas
    range(amountLayers - 2).forEach(x => {
      let deltal = []
      let sumatoria = 0

      range(this.layers[amountLayers - 2 - x].neurons).forEach(k => {
        range(this.layers[amountLayers - 3 - x].neurons).forEach(c => {
          range(this.layers[amountLayers - 1 - x].neurons).forEach(s => {
            sumatoria += deltam[s] * this.layers[amountLayers - 1 - x].weights[s][k]
          })
        })

        deltal.push(this.layers[amountLayers - 2 - x].outputs[k] * (1 - this.layers[amountLayers - 2 - x].outputs[k]) * sumatoria)

        range(this.layers[amountLayers - 3 - x].neurons).forEach(n => {
          this.layers[amountLayers - 2 - x].weights[k][n] -= this.alpha * deltal[k] * this.layers[amountLayers - 3 - x].outputs[n]
        })

        this.layers[amountLayers - 2 - x].umbrals[k] -= this.alpha * deltal[k]
      })
      deltam = deltal
    })
  }

  save() {
    // ahora guardamos nuestro layers para que el entrenamiento quede guardado
    console.log('entrenamiento finalizado...guardamos los resultados')
    writeFileSync(this.saveurl, JSON.stringify(this.layers, null, 2))
  }
}

const network = new Perceptron()
network.init()
