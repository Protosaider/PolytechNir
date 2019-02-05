import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, StandardScaler, MinMaxScaler}
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{Vector, Vectors, SparseVector}
import org.apache.spark.mllib.feature.{ChiSqSelector}
import org.apache.spark.sql.{DataFrame, SparkSession, Row}
import org.apache.spark.sql.types._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD, LogisticRegressionModel, LogisticRegressionWithLBFGS, NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
//Scala standard library doesn’t contain any classes to write files
//import java.io.{File, PrintWriter}
import java.io._

//import org.apache.spark.ml.feature.MinMaxScaler
//import org.apache.spark.ml.classification.KNNClassifier
//import org.apache.spark.mllib.knn


object Nir {
    def main(args: Array[String]) {
        val sparkConf = new SparkConf().setAppName("kdd").setMaster("local[4]")
        val sc = SparkContext.getOrCreate(sparkConf)
        val spark = SparkSession.builder().getOrCreate()

	    import spark.implicits._

        //var currentDataset: String = "kdd"
        //var currentTrainDataset: String = "trainNormal"
        //var currentTestDataset: String = "testNormal"
        //var withChi: Boolean = true
        //var chiSqNum: Integer = 25

        //wholeIteration(currentDataset, currentTrainDataset, currentTestDataset, withChi, chiSqNum, 4, 40, 2)
        wholeIteration("kdd", "trainNormal", "testNormal", true, 25,    4, 40, 2)
        wholeIteration("kdd", "trainPercent", "testPercent", true, 25,    4, 40, 2)
        wholeIteration("banknote", "banknote", "banknote", true, 25,    2, 4, 1)
        
    }

    def wholeIteration(currentDataset: String, currentTrainDataset: String, currentTestDataset: String, withChi: Boolean, chiSqNum: Integer, fromIter: Integer = 2, toIter: Integer = 4, step: Integer = 1) {

        val rddHead = loadRdd(currentDataset)
        val schema = createSchema(currentDataset, rddHead)

        val df = createDf(currentDataset, currentTrainDataset, schema)
        val pipeline = createPipeline(currentDataset, df)

        val file = new File("C://Users//Protosaider//Documents//Worktable//5k//НИР//data//Output.txt")
        val fileLog = new File("C://Users//Protosaider//Documents//Worktable//5k//НИР//data//OutputLog.txt")
        val fileBay = new File("C://Users//Protosaider//Documents//Worktable//5k//НИР//data//OutputBay.txt")
        //val pw = new PrintWriter(file)
        //val pwLog = new PrintWriter(fileLog)
        //val pwBay = new PrintWriter(fileBay)
        var message: String = "Dataset: " + currentDataset.toString + "\nTrain dataset: " + currentTrainDataset.toString + "; Test dataset: " + currentTestDataset.toString + "\n"
        //pw.write(message)
        //pwLog.write(message)
        //pwBay.write(message)
        writeStringToFile(file, message, true)
        writeStringToFile(fileLog, message, true)
        writeStringToFile(fileBay, message, true)

        var i = fromIter
        //var hasFirst: Boolean = false

        while (i <= toIter) {
            val rdd = createRdd(withChi, i, df, pipeline)
            val splits: (Array[org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint]]) = currentDataset match { 
                case "banknote" => {
                    rdd.randomSplit(Array(0.6, 0.4), seed = 11L)
                }
                case _ => {
                    Array(rdd)
                }
            }
            val training = currentDataset match {
                case "kdd" => {
                    rdd.cache()
                }
                case "banknote" => {
                    splits(0).cache()
                }
            }

            val test = currentDataset match {
                case "kdd" => {
                    val df2 = createDf(currentDataset, currentTestDataset, schema)
                    val rdd2 = createRdd(withChi, i, df2, pipeline)
                    //val rdd2 = createRdd(false, chiSqNum, df2, pipeline)
                    rdd2
                }
                case "banknote" => {
                    splits(1)
                }
            }

            val numIterationsSVM = 100
            val model = SVMWithSGD.train(training, numIterationsSVM)
            model.clearThreshold() // Clear the default threshold.
            val modelled = model.predict(test.map(vec => vec.features))
            val score = modelled.zip(test.map(vec => vec.label))

            val modelLog = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)
            modelLog.clearThreshold() // Clear the default threshold.
            val modelledLog = modelLog.predict(test.map(vec => vec.features))
            val scoreLog = modelledLog.zip(test.map(vec => vec.label))		

            val modelBay = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
            val modelledBay = modelBay.predict(test.map(vec => vec.features))
            val scoreBay = modelledBay.zip(test.map(vec => vec.label))
            val accuracy = 1.0 * scoreBay.filter(x => x._1 == x._2).count() / test.count()

            //printMetrics(score, pw, i)
            //printMetrics(scoreLog, pwLog, i)
            //printMetrics(scoreBay, pwBay, i)
            
            printMetrics(score, file, i)
            printMetrics(scoreLog, fileLog, i)
            printMetrics(scoreBay, fileBay, i)
            i = i + step
        }

        //pw.close
        //pwLog.close
        //pwBay.close
    }

    def createPipeline(file: String, df: org.apache.spark.sql.DataFrame): (org.apache.spark.ml.Pipeline) = {
        file match {
            case "kdd" => {
                val indexerProtoType = new StringIndexer().setInputCol("protocol_type").setOutputCol("protocol_typeIndexed")
                val indexerService = new StringIndexer().setInputCol("service").setOutputCol("serviceIndexed")
                val indexerFlag = new StringIndexer().setInputCol("flag").setOutputCol("flagIndexed")
                val assembleCols = Set(df.columns: _*) -- Seq("class", "protocol_type", "service", "flag") ++ Seq("protocol_typeIndexed", "serviceIndexed", "flagIndexed")
                val assembler = new VectorAssembler().setInputCols(assembleCols.toArray).setOutputCol("featureVector")
                val scaler = new StandardScaler().setInputCol("featureVector").setOutputCol("scaledFeatureVector").setWithStd(true).setWithMean(false)
                val indexer = new StringIndexer().setInputCol("class").setOutputCol("label")
                ( new Pipeline().setStages(Array(indexerProtoType, indexerService, indexerFlag, assembler, scaler, indexer)) )
            } 
            case "banknote" => {
                val assembleCols = Set(df.columns: _*) -- Seq("class")
                val assembler = new VectorAssembler().setInputCols(assembleCols.toArray).setOutputCol("featureVector")
                    //val normalizer = new Normalizer().setInputCol("featureVector").setOutputCol("scaledFeatureVector").setP(1.0)
                //val normalizer = new Normalizer().setInputCol("featureVector").setOutputCol("normFeatures").setP(1.0)
	// val normalizer = new Normalizer().setInputCol("featureVector").setOutputCol("normFeatures").setP(Double.PositiveInfinity)
                    //val scaler = new StandardScaler().setInputCol("normFeatures").setOutputCol("scaledFeatureVector").setWithStd(true).setWithMean(false)
val scaler = new MinMaxScaler().setInputCol("featureVector").setOutputCol("scaledFeatureVector")
                val indexer = new StringIndexer().setInputCol("class").setOutputCol("label")
                //( new Pipeline().setStages(Array(assembler, normalizer, indexer)) )
                ( new Pipeline().setStages(Array(assembler, scaler, indexer)) )
            }
        }
    }

    def loadRdd(file: String): (org.apache.spark.rdd.RDD[String]) = {
        val rddHeadMap = scala.collection.mutable.Map[String, String]()
        rddHeadMap += ("kdd" -> "file:///Users//Protosaider//Documents//Worktable//5k//НИР//data//kddcup.corrected.names.txt")
        rddHeadMap += ("banknote" -> "file:///Users//Protosaider//Documents//Worktable//5k//НИР//data//data_banknote_head.txt")     
        ( sc.textFile(rddHeadMap(file)) )
    }

    def createSchema(file: String, rddHead: org.apache.spark.rdd.RDD[String]): (org.apache.spark.sql.types.StructType) = {
        //val rddHead = loadRdd(file)
        val head = rddHead.map(line => line.split(","))
        val fields : Array[org.apache.spark.sql.types.StructField] = file match {  // match prevents return of type Any
            case "kdd" => {
                head.map(line => StructField(line(0), if (line(0) == "protocol_type" || line(0) == "service" || line(0) == "flag" || line(0) == "class") StringType else DoubleType, nullable = false)).collect.toArray
            } 
            case "banknote" => {
                head.map(line => StructField(line(0), DoubleType, nullable = false)).collect.toArray
            }
        }
        val features = head.map(line => line(0)).filter(s => s != "class").collect.toArray
        ( org.apache.spark.sql.types.StructType(fields) )
    }

    def createDf(file: String, dfSource: String = "trainNormal", schema: org.apache.spark.sql.types.StructType): (org.apache.spark.sql.DataFrame) = {     
        //val schema = createSchema(file)
        val df : org.apache.spark.sql.DataFrame = file match {
            case "kdd" => {
                val unclean : org.apache.spark.sql.DataFrame = dfSource match {
                    case "trainPercent" => {
                        spark.read.schema(schema).csv("file:///Users//Protosaider//Documents//Worktable//5k//НИР//data//KDDTrain+_20Percent.txt")
                    } 
                    case "trainNormal" => {
                        spark.read.schema(schema).csv("file:///Users//Protosaider//Documents//Worktable//5k//НИР//data//KDDTrain+.txt")
                    }
                    case "testPercent" => {
                        spark.read.schema(schema).csv("file:///Users//Protosaider//Documents//Worktable//5k//НИР//data//KDDTest-21.txt")
                    }
                    case "testNormal" => {
                        spark.read.schema(schema).csv("file:///Users//Protosaider//Documents//Worktable//5k//НИР//data//KDDTest+.txt")
                    }
                }
                val dfBinaryClass = unclean.withColumn("class", when(col("class") === "normal", col("class")).otherwise("attack"))
                dfBinaryClass.drop("level")
            } 
            case "banknote" => {
                spark.read.schema(schema).csv("file:///Users//Protosaider//Documents//Worktable//5k//НИР//data//data_banknote_authentication.txt")
            }
        }
        ( df )
    }

    def createRdd(withChi: Boolean, chiSqNum: Integer, df: org.apache.spark.sql.DataFrame, pipeline: org.apache.spark.ml.Pipeline): (org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint]) = {
        val dataset = pipeline.fit(df)
	    val rddata = dataset.transform(df).select("label", "scaledFeatureVector").as[(Double, org.apache.spark.ml.linalg.SparseVector)].map(row => LabeledPoint(row._1, Vectors.fromML(row._2))).rdd
        var rdd = if (withChi) {
		    val selector = new ChiSqSelector(chiSqNum)
		    val transformer = selector.fit(rddata)
		    val rddFeaturesTransformed = transformer.transform(rddata.map(vec => vec.features))
		    rddata.map(vec => vec.label).zip(rddFeaturesTransformed).map(vec => LabeledPoint(vec._1, vec._2))
	    } else {
		    rddata
	    }
        (rdd)
    }

    //def printMetrics(score: org.apache.spark.rdd.RDD[(Double, Double)], writer: java.io.PrintWriter, chiSqNum: Integer) {
    //    val metrics = new BinaryClassificationMetrics(score)
    //    val auROC = metrics.areaUnderROC
    //    val auPRC = metrics.areaUnderPR
    //    //writer.write("chiSqNum: " + chiSqNum.toString + " " + auROC.toString + " " + auPRC.toString + "\n")
    //}

    def printMetrics(score: org.apache.spark.rdd.RDD[(Double, Double)], file: File, chiSqNum: Integer) {
        val metrics = new BinaryClassificationMetrics(score)
        val auROC = metrics.areaUnderROC
        val auPRC = metrics.areaUnderPR
        //writer.write("chiSqNum: " + chiSqNum.toString + " " + auROC.toString + " " + auPRC.toString + "\n")
        writeStringToFile(file, "chiSqNum: " + chiSqNum.toString + " " + auROC.toString + " " + auPRC.toString + "\n", true)
    }

    def using[A <: {def close() : Unit}, B](resource: A)(f: A => B): B = try f(resource) finally resource.close()

    def writeStringToFile(file: File, data: String, appending: Boolean = false) = using(new FileWriter(file, appending))(_.write(data))
 
}

        val sparkConf = new SparkConf().setAppName("kdd").setMaster("local[4]")
        val sc = SparkContext.getOrCreate(sparkConf)
        val spark = SparkSession.builder().getOrCreate()

	    import spark.implicits._

        val df = createDf("kdd", "train")
        val pipeline = createPipeline("kdd")
        val dataset = pipeline.fit(df)
	    val rddata = dataset.transform(df).select("label", "scaledFeatureVector").as[(Double, org.apache.spark.ml.linalg.SparseVector)].map(row => LabeledPoint(row._1, Vectors.fromML(row._2))).rdd
        val rdd = if (withChi) {
		    val selector = new ChiSqSelector(chiSqNum)
		    val transformer = selector.fit(rddata)
		    val rddFeaturesTransformed = transformer.transform(rddata.map(vec => vec.features))
		    rddata.map(vec => vec.label).zip(rddFeaturesTransformed).map(vec => LabeledPoint(vec._1, vec._2))
	    } else {
		    rddata
	    }
        
        val splits: () = file match { 
            case "banknote" => {
                rdd.randomSplit(Array(0.6, 0.4), seed = 11L)
            }
        }

        val training = file match {
            case "kdd" => {
                rdd.cache()
            }
            case "banknote" => {
                splits(0).cache()
            }
        }
            

        val test = file match {
            case "kdd" => {
                val schema = createSchema(file)

                val unclean2 = spark.read.schema(schema).csv("file:///Users//Protosaider//Documents//Worktable//5k//НИР//data//KDDTest+.txt")

                val dfBinaryClass2 = unclean2.withColumn("class", when(col("class") === "normal", col("class")).otherwise("attack"))
                val df2 = dfBinaryClass2.drop("level")

                val dataset2 = pipeline.fit(df2)
                val rddata2 = dataset2.transform(df2).select("label", "scaledFeatureVector").as[(Double, org.apache.spark.ml.linalg.SparseVector)].map(row => LabeledPoint(row._1, Vectors.fromML(row._2))).rdd

                val rdd2 = if (withChi) {
                    val selector = new ChiSqSelector(35)
                    val transformer = selector.fit(rddata2)
                    val rddFeaturesTransformed = transformer.transform(rddata2.map(vec => vec.features))
                    rddata2.map(vec => vec.label).zip(rddFeaturesTransformed).map(vec => LabeledPoint(vec._1, vec._2))
                } else {
                    rddata2
                }
                rdd2
            }
            case "banknote" => {
                splits(1)
            }
        }

        val numIterationsSVM = 100
    val model = SVMWithSGD.train(training, numIterationsSVM)
        model.clearThreshold() // Clear the default threshold.
        val modelled = model.predict(test.map(vec => vec.features))
        val score = modelled.zip(test.map(vec => vec.label))

    val modelLog = new LogisticRegressionWithLBFGS().setNumClasses(2).run(training)
        modelLog.clearThreshold() // Clear the default threshold.
        val modelledLog = modelLog.predict(test.map(vec => vec.features))
        val scoreLog = modelledLog.zip(test.map(vec => vec.label))		

    val modelBay = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
        val modelledBay = modelBay.predict(test.map(vec => vec.features))
        val scoreBay = modelledBay.zip(test.map(vec => vec.label))
        val accuracy = 1.0 * scoreBay.filter(x => x._1 == x._2).count() / test.count()


    val metrics = new BinaryClassificationMetrics(score)
        val auROC = metrics.areaUnderROC
        val auPRC = metrics.areaUnderPR
        // val fMesureBeta1 = metrics.fMeasureByThreshold

    val metricsLog = new BinaryClassificationMetrics(scoreLog)
        val auROCLog = metricsLog.areaUnderROC
        val auPRCLog = metricsLog.areaUnderPR
      
    val metricsBay = new BinaryClassificationMetrics(scoreBay)
        val auROCBay = metricsBay.areaUnderROC
        val auPRCBay = metricsBay.areaUnderPR


        