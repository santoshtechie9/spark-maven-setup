package com.v2maestros.bda.scala.practice
/*
-----------------------------------------------------------------------------

                   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Practice Solutions : Spark with Scala - Operations
-----------------------------------------------------------------------------
*/

object SparkOperationsPracticeSolutions extends App{
	
	import com.v2maestros.bda.scala.common._
  	import org.apache.log4j.Logger
	import org.apache.log4j.Level;
	
	Logger.getLogger("org").setLevel(Level.ERROR);
	Logger.getLogger("akka").setLevel(Level.ERROR);
  	
  	val spContext = SparkCommonUtils.spContext
  	val datadir = SparkCommonUtils.datadir
  	
  	/*-----------------------------------------------------------------------------
	1. Your course resource has a CSV file "iris.csv". 
	Load that file into an RDD called irisRDD
	Cache the RDD and count the number of lines
	-----------------------------------------------------------------------------*/
	val irisData = spContext.textFile(datadir + "/iris.csv")
	irisData.cache()
	//Loads only now.
	irisData.count()
	irisData.first()
	irisData.take(5)
	
	/*-----------------------------------------------------------------------------
	2. Filter irisRDD for lines that contain "versicolor" and count them.
	-----------------------------------------------------------------------------*/
	val versiData=irisData.filter(x => x.contains("versicolor"))
	println("Total versicolor = " + versiData.count())
	
	/*-----------------------------------------------------------------------------
	3. Find the average Sepal.Length for all flowers in the irisRDD
	
	Note: Regular expression for checking a float is 
	"[+-]?(([1-9][0-9]*)|(0))([.,][0-9]+)?"
	-----------------------------------------------------------------------------*/
	
	//Function to check if a number is float
	def isAllDigits(x: String) = x.matches("[+-]?(([1-9][0-9]*)|(0))([.,][0-9]+)?")
	
	//Function to perform reduce - get and summarize Sepal Length values
	def getSepalLength( irisStr : String) : String= {
	    if ( isAllDigits(irisStr)) {
	        return irisStr
	    }
	    else {
	        val attList = irisStr.split(",")
	        if ( isAllDigits(attList(0) )) {
	            return attList(0)
	        }
	        else {
	            return "0"
	        }
	    }
	}
	
	//find average Sepal Length   
	val totSepLen =  irisData.reduce((x,y) =>  (getSepalLength(x).toFloat 
			+ getSepalLength(y).toFloat).toString )
	val avgSepLen=totSepLen.toFloat/(irisData.count()-1)
	
	println("Avg Sepal Length : " + avgSepLen)
	
	/*-----------------------------------------------------------------------------
	4. Find the number of records in irisRDD, whose Sepal.Length is 
	greater than the Average Sepal Length we found in the earlier practice
	
	Note: Use Broadcast and Accumulator variables for this practice
	-----------------------------------------------------------------------------*/
	//Initialize accumulator
	val sepalHighCount = spContext.longAccumulator
	
	//Set Broadcast variable
	val avgSepalLen=spContext.broadcast(avgSepLen)
	
	//Write a function to do the compare and count
	def findHighLen(line : String) : String  = {
	
	    val attList=line.split(",")
	    if ( isAllDigits(attList(0)) ) {
	        
	        if ( attList(0).toFloat > avgSepalLen.value ) {
	            sepalHighCount.add(1)
	            return "high"
	        }
	        else {
	            return "low"
	        }
	    }
	    else {
	        return "Error"
	    }
	   
	}
	//do the map
	val sumData=irisData.map(findHighLen)
	sumData.count()
	//Print the result
	println("Sepal High Count = " + sepalHighCount.value)
	
	/*-----------------------------------------------------------------------------
	Hope you had some good practice !! Recommend trying out your own use cases
	-----------------------------------------------------------------------------*/
}