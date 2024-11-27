package sgbit.cse
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.rand
object Main {
  def main(args: Array[String]): Unit = {
    // Initialize SparkSession

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.apache.hadoop").setLevel(Level.WARN)
    Logger.getLogger("org.apache.kafka").setLevel(Level.WARN)

    // Optionally suppress root logger
    Logger.getRootLogger.setLevel(Level.WARN)
    val spark = SparkSession.builder()
      .appName("Amazon Reviews Sentiment Analysis")
      .master("local[*]")
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.3.0")
      .getOrCreate()

    import spark.implicits._

    // Load the dataset
    val filePath = "amazon_reviews.csv" // Replace with the actual file path
    val rawData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(filePath)


    // Select necessary columns and cast `label` to Double
    val data = rawData.select("review_id", "review_text", "label")
      .withColumn("label", $"label".cast("Double"))

    // Filter out rows with null or NaN labels
    val cleanedData = data.filter($"label".isNotNull && $"label".isNaN === false)

    // NLP pipeline for text preprocessing
    val documentAssembler = new DocumentAssembler()
      .setInputCol("review_text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols("token")
      .setOutputCol("normalized")

    val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("normalized")
      .setOutputCol("cleanTokens")

    // Finisher to convert structured tokens to plain strings
    val finisher = new Finisher()
      .setInputCols("cleanTokens")
      .setOutputCols("cleanedTokensArray")
      .setOutputAsArray(true)

    // Convert tokenized text into a numerical format using CountVectorizer
    val countVectorizer = new CountVectorizer()
      .setInputCol("cleanedTokensArray")
      .setOutputCol("featuresVector")

    // Naive Bayes Classifier
    val naiveBayes = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("featuresVector")

    // Create a pipeline
    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        normalizer,
        stopWordsCleaner,
        finisher,
        countVectorizer,
        naiveBayes
      ))

    // Train-test split
    val Array(trainingData, testData) = cleanedData.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Train the model
    val model = pipeline.fit(trainingData)

    // Make predictions
    val predictions = model.transform(testData)

    // Evaluate the model
    val evaluatorAccuracy = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val evaluatorPrecision = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")

    val evaluatorRecall = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")

    val accuracy = evaluatorAccuracy.evaluate(predictions)
    val precision = evaluatorPrecision.evaluate(predictions)
    val recall = evaluatorRecall.evaluate(predictions)

//    println(s"Model Performance Metrics:")
    println(f"Accuracy: $accuracy%1.2f")
    println(f"Precision: $precision%1.2f")
    println(f"Recall: $recall%1.2f")

    val randomPredictions = predictions
      .select("review_id", "review_text", "label", "prediction")
      .orderBy(rand()) // Randomly order the rows
      .limit(20) // Limit the result to 20 rows

    randomPredictions.show(truncate = false)
    // Show predictions
    //predictions.select("review_id", "review_text", "label", "prediction").show(truncate = false)

    // Stop the SparkSession
    spark.stop()
  }
}
