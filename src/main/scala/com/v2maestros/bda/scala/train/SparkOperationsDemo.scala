/*-----------------------------------------------------------------------------

                   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Operations Examples 
-----------------------------------------------------------------------------
*/

package com.v2maestros.bda.scala.train

object SparkOperationsDemo extends App {

  	import com.v2maestros.bda.scala.common._
  	import org.apache.log4j.Logger
	import org.apache.log4j.Level;
	
	Logger.getLogger("org").setLevel(Level.ERROR);
	Logger.getLogger("akka").setLevel(Level.ERROR);
  	
  	val sc = SparkCommonUtils.spContext
	/*............................................................................
	////   Loading Data From a Collection
	//............................................................................*/
	val collData = sc.parallelize(Array(3, 5, 4, 7, 4))
	collData.cache()
	collData.count()

	/*............................................................................
	////   Loading and Storing Data
	............................................................................*/

	//Load the file. Lazy initialization
	val autoAllData = sc.textFile(SparkCommonUtils.datadir 
			+ "/auto-data.csv")
	autoAllData.cache()
	//Loads only now.
	autoAllData.count()
	autoAllData.first()
	autoAllData.take(5)

	//Storing Data to files
	autoAllData.saveAsTextFile(SparkCommonUtils.datadir 
			+ "auto-data-updated.csv")
			
	//Converting to scala lists
	val autoList = autoAllData.collect()
	
	for (x <- autoList) { println(x) }

	//............................................................................
	////   Transformations
	//............................................................................

	//Map and create a new RDD
	var tsvData = autoAllData.map(x => x.replace(",", "\t"))
	tsvData.take(5)
	
	//Remove the first header line
	val header1 = autoAllData.first
	val autoData = autoAllData.filter( s => !s.equals(header1))
	autoData.count()

	//Filter only for toyota cars and create a new RDD
	var toyotaData = autoAllData.filter(x => x.contains("toyota"))
	toyotaData.count()
	
	// Distinct Example
	collData.distinct().collect()

	//FlatMap with inline function
	var words = toyotaData.flatMap(x => x.split(","))
	words.take(20)
	words.count()

	//cleanse and transform an RDD with a function
	def cleanseRDD(autoStr: String): String = {
		
		val attList = autoStr.split(",")
		
		if (attList(3).matches("two")) {
			attList(3) = "2"
		} else {
			attList(3) = "4"
		}

		//convert drive to upper case
		attList(5) = attList(5).toUpperCase()
		attList.mkString(",")
	}

	val cleanedData = autoData.map(cleanseRDD)
	cleanedData.collect()

	//Set operations
	val words1 = sc.parallelize(Array("hello", "war", "peace", "world"))
	val words2 = sc.parallelize(Array("war", "peace", "universe"))

	words1.union(words2).distinct().collect()
	words1.intersection(words2).collect()

	/*............................................................................
	////   Actions
	//............................................................................*/

	//reduce
	collData.reduce((x, y) => x + y)
	
	//find the shortest line
	autoData.reduce((x, y) => if (x.length() < y.length) x else y)


	def isAllDigits(x: String) = x.matches("^\\d*$")

	//Use a function to extract MPG 
	def getMPG(autoStr: String): String = {
		if (isAllDigits(autoStr)) {
			return autoStr
		} else {
			val attList = autoStr.split(",")
			if (isAllDigits(attList(9))) {
				return attList(9)
			} else {
				return "0"
			}
		}
	}

	//find average MPG-City for all cars    
	val totMPG = autoData.reduce((x, y) 
			=> (getMPG(x).toInt + getMPG(y).toInt).toString)
			
	totMPG.toInt / (autoData.count())

	//............................................................................
	////   Working with Key/Value RDDs
	//............................................................................

	//create a KV RDD of auto Brand and Horsepower
	val cylHPData = autoData.map(x 
			=> (x.split(",")(0), x.split(",")(7)))
	cylHPData.take(5)
	cylHPData.keys.distinct.collect()


	//Add a count 1 to each record and thn reduce to find totals of HP and counts
	val brandValues = cylHPData.mapValues(x => (x.toInt, 1))
	brandValues.collect()

	val totValues = brandValues.reduceByKey((x, y) 
			=> (x._1 + y._1, x._2 + y._2))
	totValues.collect()
	
	//find average by dividing HP total by count total
	totValues.mapValues(x => x._1 / x._2).collect()

	//............................................................................
	////   Advanced Spark : Accumulators & Broadcast Variables
	//............................................................................

	//function that splits the line as well as counts sedans and hatchbacks
	//Speed optimization

	//Initialize accumulator
	val sedanCount = sc.longAccumulator;
	val hatchbackCount = sc.longAccumulator;

	//Set Broadcast variable
	val sedanText = sc.broadcast("sedan")
	val hatchbackText = sc.broadcast("hatchback")

	def splitLines(line: String): Array[String] = {
		if (line.contains(sedanText.value)) {
			sedanCount.add(1)
		}
		if (line.contains(hatchbackText.value)) {
			hatchbackCount.add(1)
		}
		line.split(",")
	}
	//do the map
	val splitData = autoData.map(splitLines)

	//Make it execute the map (lazy execution)
	splitData.count()
	println(sedanCount.value + "  " + hatchbackCount.value)

	//............................................................................
	////   Advanced Spark : Partitions
	//............................................................................
	collData.getNumPartitions

	//Specify no. of partitions.
	val collData2 = sc.parallelize(Array(3, 5, 4, 7, 4), 4)

	println(collData2.getNumPartitions)
	collData2.cache()

	//localhost:4040 shows the current spark instance

}