package com.v2maestros.bda.scala.train

object testing1 {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(101); 
  println("Welcome to the Scala worksheet")
  
  import com.v2maestros.bda.scala.common._;$skip(86); 
  
  val sc = SparkConnection.spContext;System.out.println("""sc  : org.apache.spark.SparkContext = """ + $show(sc ));$skip(51); 
  
  val collData=sc.parallelize(Array(3,5,4,7,4));System.out.println("""collData  : org.apache.spark.rdd.RDD[Int] = """ + $show(collData ));$skip(18); val res$0 = 
	collData.cache();System.out.println("""res0: com.v2maestros.bda.scala.train.testing1.collData.type = """ + $show(res$0));$skip(18); val res$1 = 
	collData.count();System.out.println("""res1: Long = """ + $show(res$1))}
  
}
