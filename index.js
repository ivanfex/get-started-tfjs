const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')
const irisSample = require('./irisSample')
const irisTest = require('./irisTest')

const flowerData = tf.tensor2d(irisSample.map(flower => [
	flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width,
]))

const flowerSpecies = tf.tensor2d(irisSample.map(flower => [
	flower.species === "setosa" ? 1 : 0,
	flower.species === "virginica" ? 1 : 0,
	flower.species === "versicolor" ? 1 : 0,
]))

const flowersToPredict = tf.tensor2d(irisTest.map(flower => [
	flower.sepal_length, flower.sepal_width, flower.petal_length, flower.petal_width,
]))

const botanic = tf.sequential()

botanic.add(tf.layers.dense({
	inputShape: [4],
	activation: "sigmoid",
	units: 5
}))

botanic.add(tf.layers.dense({
	inputShape: [5],
	activation: "sigmoid",
	units: 3
}))

botanic.add(tf.layers.dense({
	activation: "sigmoid",
	units: 3
}))

botanic.compile({
	loss: "meanSquaredError",
	optimizer: tf.train.adam(.06)
})

botanic.fit(flowerData, flowerSpecies, {epochs: 100})
	.then((history) => {
		console.log(history)
		botanic.predict(flowersToPredict).print()
	})
	.catch(console.error)