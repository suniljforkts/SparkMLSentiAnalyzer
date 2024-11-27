// Name of the project
name := "Test"

// Scala version
scalaVersion := "2.12.15" // Ensure this matches your Spark version

// Spark and Spark NLP versions
val sparkVersion = "3.5.2"
val sparkNLPVersion = "4.3.0"

// Resolvers for dependencies
resolvers += "Spark Packages Repo" at "https://repos.spark-packages.org/"
mainClass in Compile := Some("Main")
// Dependencies
libraryDependencies ++= Seq(
  // Spark Core and SQL
  "org.apache.spark" %% "spark-core" % "3.5.2",
  "org.apache.spark" %% "spark-sql" % "3.5.2",

  // Spark MLlib for machine learning and pipelines
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  // Spark NLP
  "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNLPVersion,
  // Logging
  "org.slf4j" % "slf4j-api" % "1.7.36",
  "org.slf4j" % "slf4j-log4j12" % "1.7.36"
)
