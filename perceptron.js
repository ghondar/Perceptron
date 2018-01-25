const { question } = require('readline-sync')
const { readFileSync, writeFileSync } = require('fs')

class Perceptron {
  constructor() {
    this.alpha = 0.6 // factor de aprendizaje. Cuanto mayo es, antes aprende pero tiene menor precision
    this.perceptron = [] // array que nos define el perceptron y por cada capa contiene un subarray de la siguiente forma: [numero de neuronas de la capa, [ umbrales de la actuacion(u) ], [ pesos(w) ], [salidas de las neuronas(a) ]]
    this.ntrain = 0 // numero de entrenamientos ejecutados
    this.saveurl = __dirname + '/perceptron.txt' // url donde guardaremos la red neuronal una vez entrenada
    this.trainingdata = __dirname + '/entrenamientoperc.txt' // url donde tenemos la training data
  }

  init() {
    this.build() // construimos la red neuronal
    this.train() // entrenamos la red neuronal

    while (true) { // utilizamos la red neuronal
      for (let i in [ ...Array(this.perceptron[0][0]).keys() ]) {
        i = parseInt(i)
        this.perceptron[0][3][i] = parseFloat(question('entrada ' + i + ' : '))
      }
      this.percep()
    }
  }

  build() { // funcion que contruye la red neuronal
    console.log('***PERCEPTRON***')
    for (let i in [ ...Array(parseFloat(question('numero de capas: '))).keys() ]) {
      i = parseInt(i)
      let capa = []
      let u = []
      let w = []
      let a = []
      capa.push(parseFloat(question('numero de neuronas en la capa ' + i + ' : ')))

      for (let j in [ ...Array(capa[0]).keys() ]) {
        j = parseInt(j)
        u.push(Math.random()) // Creamos umbrales de actuacion aleatorios
        a.push(Math.random()) // Creamos salidas aleatorios
        let wc = []
        if(i != 0)
          for (let k in [ ...Array(this.perceptron[i - 1][0]).keys() ]) {
            k = parseInt(k)
            wc.push(Math.random()) // creamos pesos aleatorios
          }

        w.push(wc)
      }

      capa.push(u)
      capa.push(w)
      capa.push(a)
      this.perceptron.push(capa) // anadimos la capa a la red neuronal
    }
  }

  percep() { // funcion que calcula la salida de la red neuronal ante una serie de entradas
    console.log('salida de las neuronas capa 0:')
    console.log(this.perceptron[0][3])
    for (let k in [ ...Array(this.perceptron.length - 1).keys() ]) {
      k = parseInt(k)
      for (let i in [ ...Array(this.perceptron[k + 1][0]).keys() ]) {
        i = parseInt(i)
        let suma = 0
        for (let j in [ ...Array(this.perceptron[k][0]).keys() ]) {
          j = parseInt(j)
          suma = suma + (this.perceptron[k][3][j] * this.perceptron[k + 1][2][i][j])
        }
        suma = suma + this.perceptron[k + 1][1][i]
        suma = (1 / (1 + Math.exp(-suma)))
        this.perceptron[k + 1][3][i] = suma
      }
      console.log('salida de las neuronas capa ' + (k + 1) + ' : ')
      console.log(this.perceptron[k + 1][3])
    }
  }

  train() {
    console.log('-+-Comienza el entrenamiento del perceptron-+-')
    for (let r in [ ...Array(parseFloat(question('numero de veces que quiere repetir el entrenamiento: '))).keys() ])
      this.readlines()

    this.save()
  }

  readlines() {
    let entrenamiento = readFileSync(this.trainingdata, 'utf8')
    let lineas = entrenamiento.split('\n')
    for (let i in [ ...Array(lineas.length).keys() ]) {
      i = parseInt(i)
      this.ntrain += 1
      console.log('***entrenamiento numero ' + this.ntrain + ' ***')

      let linea = lineas[i]
      let boolean = false
      let numero = ''
      let k = 0
      let se = []

      for (let x in [ ...Array(this.perceptron[this.perceptron.length - 1][0]).keys() ]) {
        x = parseInt(x)
        se.push(0)
      }

      for (let j in [ ...Array(linea.length).keys() ]) {
        j = parseInt(j)
        if(linea[j] == ']' && boolean) {
          se[k] = numero
          numero = ''
          k = 0
        }

        if(linea[j] == ']' && boolean === false) {
          boolean = true
          this.perceptron[0][3][k] = parseFloat(numero)
          numero = ''
          k = 0
        }

        if(linea[j] != ',' && linea[j] != '[' && linea[j] != ']')
          numero = numero + linea[j]

        if(linea[j] == ',' && boolean == false) {
          this.perceptron[0][3][k] = parseFloat(numero)
          numero = ''
          k += 1
        }

        if(linea[j] == ',' && boolean) {
          se[k] = parseFloat(numero)
          numero = ''
          k += 1
        }
      }

      se[this.perceptron[this.perceptron.length - 1][0] - 1] = parseFloat(se[this.perceptron[this.perceptron.length - 1][0] - 1])
      console.log('entrada: ' + JSON.stringify(this.perceptron[0][3]) + ' salida esperada: ' + JSON.stringify(se))
      this.percep()
      this.backpropagation(se)
    }
  }

  backpropagation(se) {
    // Calculamos los errores y realizamos las correciones oportunas
    // empezamos por la ultima capa de neuronas

    let deltam = []
    for (let k in [ ...Array(this.perceptron[this.perceptron.length - 1][0]).keys() ]) {
      k = parseInt(k)
      deltam.push(this.perceptron[this.perceptron.length - 1][3][k] * (1 - this.perceptron[this.perceptron.length - 1][3][k]) * (this.perceptron[this.perceptron.length - 1][3][k] - se[k]))
      for (let n in [ ...Array(this.perceptron[this.perceptron.length - 2][0]).keys() ]) {
        n = parseInt(n)
        this.perceptron[this.perceptron.length - 1][2][k][n] = this.perceptron[this.perceptron.length - 1][2][k][n] - this.alpha * deltam[k] * this.perceptron[this.perceptron.length - 2][3][n]
      }
      this.perceptron[this.perceptron.length - 1][1][k] = this.perceptron[this.perceptron.length - 1][1][k] - this.alpha * deltam[k]
    }

    // seguimos con las capas ocultas de neuronas
    for (let x in [ ...Array(this.perceptron.length - 2).keys() ]) {
      x = parseInt(x)
      let deltal = []
      let sumatorio2 = 0

      for (let k in [ ...Array(this.perceptron.length - 2).keys() ]) {
        k = parseInt(k)
        for (let c in [ ...Array(this.perceptron[this.perceptron.length - 3 - x][0]).keys() ]) {
          c = parseInt(c)
          for (let s in [ ...Array(this.perceptron[this.perceptron.length - 1 - x][0]).keys() ]) {
            s = parseInt(s)
            sumatorio2 = sumatorio2 + deltam[s] * this.perceptron[this.perceptron.length - 1 - x][2][s][k]
          }
        }
        deltal.push((this.perceptron[this.perceptron.length - 2 - x][3][k] * (1 - this.perceptron[this.perceptron.length - 2 - x][3][k])) * sumatorio2)
        for (let n in [ ...Array(this.perceptron[this.perceptron.length - 3 - x][0]).keys() ]) {
          n = parseInt(n)
          this.perceptron[this.perceptron.length - 2 - x][2][k][n] = this.perceptron[this.perceptron.length - 2 - x][2][k][n] - this.alpha * deltal[k] * this.perceptron[this.perceptron.length - 3 - x][3][n]
        }

        this.perceptron[this.perceptron.length - 2 - x][1][k] = this.perceptron[this.perceptron.length - 2 - x][1][k] - this.alpha * deltal[k]
      }
      deltam = deltal
    }
  }

  save() {
    // ahora guardamos nuestro perceptron para que el entrenamiento quede guardado
    console.log('entrenamiento finalizado...guardamos los resultados')
    writeFileSync(this.saveurl, JSON.stringify(this.perceptron, 2))
  }
}

const network = new Perceptron()
network.init()
