package com.test.spark;

import java.util.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.ml.*;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.*;

public class Wine {
  public static void main(String[] args) {
    /**SparkConf conf = new SparkConf().setMaster("local").setAppName("WineQualityClassification"); **/
	SparkConf conf = new SparkConf().setAppName("WineQualityClassification");
    JavaSparkContext sc = new JavaSparkContext(conf);
    
    /** LOCAL VARIABLES 
    String path_training = "C://spark/spark-2.4.7-bin-hadoop2.7/data/mllib/TrainingDataset.csv";
    String path_validation = "C://spark/spark-2.4.7-bin-hadoop2.7/data/mllib/ValidationDataset.csv";
    **/
    
    String path_training = "s3://dsqualitywine/TrainingDataset.csv";
    String path_validation = "s3://dsqualitywine/ValidationDataset.csv"; 
    
    
    JavaRDD<String> data_training = sc.textFile(path_training);
    JavaRDD<String> data_validation = sc.textFile(path_validation);
    
    /** REMOVE HEADERS **/
    final String header_training = data_training.first();
    JavaRDD<String> data1 = data_training.filter(new Function<String, Boolean>() {
        @Override
        public Boolean call(String s) throws Exception {
            return !s.equalsIgnoreCase(header_training);
        }
    });
    
    final String header_validation = data_validation.first();
    JavaRDD<String> data2 = data_validation.filter(new Function<String, Boolean>() {
        @Override
        public Boolean call(String s) throws Exception {
            return !s.equalsIgnoreCase(header_validation);
        }
    });
    
     /** CREATING TRAINING LABELEDPOINT **/    
    JavaRDD<LabeledPoint> parsedData1 = data1
            .map(new Function<String, LabeledPoint>() {
                public LabeledPoint call(String line) throws Exception {
                    String[] parts = line.split(";");
                    return new LabeledPoint(Double.parseDouble(parts[11]),
                            Vectors.dense(Double.parseDouble(parts[0]),
                                    Double.parseDouble(parts[1]),
                                    Double.parseDouble(parts[2]),
                                    Double.parseDouble(parts[3]),
                                    Double.parseDouble(parts[4]),
                                    Double.parseDouble(parts[5]),
                                    Double.parseDouble(parts[6]),
                                    Double.parseDouble(parts[7]),
                                    Double.parseDouble(parts[8]),
                                    Double.parseDouble(parts[9]),
                                    Double.parseDouble(parts[10])));
                }
            });
    
   
    /** CREATING VALIDATION LABELEDPOINT **/
    JavaRDD<LabeledPoint> parsedData2 = data2
            .map(new Function<String, LabeledPoint>() {
                public LabeledPoint call(String line) throws Exception {
                    String[] parts = line.split(";");
                    return new LabeledPoint(Double.parseDouble(parts[11]),
                            Vectors.dense(Double.parseDouble(parts[0]),
                                    Double.parseDouble(parts[1]),
                                    Double.parseDouble(parts[2]),
                                    Double.parseDouble(parts[3]),
                                    Double.parseDouble(parts[4]),
                                    Double.parseDouble(parts[5]),
                                    Double.parseDouble(parts[6]),
                                    Double.parseDouble(parts[7]),
                                    Double.parseDouble(parts[8]),
                                    Double.parseDouble(parts[9]),
                                    Double.parseDouble(parts[10])));
                }
            });
       
        
    /** BUILDING MODEL LOGISTIC REGRESSION **/
    final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
    		.setNumClasses(10)
    		.run(parsedData1.rdd());
    JavaRDD<Tuple2<Object, Object>> predictionAndLabels = parsedData1.map( p -> {
        Double prediction = model.predict(p.features());
        return new Tuple2<>(prediction, p.label());
    });
    // Get evaluation metrics.
    final MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
    System.out.println();
	System.out.println("----------------------------------------------------------------------------");
    System.out.println("Logistic Regression Validation Accuracy: " + metrics.accuracy());
    System.out.println("----------------------------------------------------------------------------");
	System.out.println();
    
    
    double f_score = metrics.weightedFMeasure();
	System.out.println();
	System.out.println("----------------------------------------------------------------------------");
	System.out.println("Validation F Measure = " + f_score);
	System.out.println("----------------------------------------------------------------------------");
	System.out.println();
    
    /**DECISION TREE**/
    /**
    int numClasses = 10;
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
    String impurity = "gini";
    int maxDepth = 5;
    int maxBins = 32;
    
    DecisionTreeModel modeltree = DecisionTree.trainClassifier(parsedData1, numClasses,categoricalFeaturesInfo, impurity, maxDepth, maxBins);

    // Predict for the test data using the model trained
    JavaPairRDD<Double, Double> predictionAndLabel = parsedData2.mapToPair(p -> new Tuple2<>(modeltree.predict(p.features()), p.label()));
    
    double accuracyDT =
            predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) parsedData2.count();
     
    System.out.println("Accuracy Decision Tree is : "+accuracyDT);
    System.out.println("Trained Decision Tree model:\n" + modeltree.toDebugString());
    **/  
    
	 /**RANDOM FOREST
	 Integer numberClasses1 = 10;
	 Map<Integer, Integer> categoricalFeaturesInfo1 = new HashMap<>();
	 Integer numTrees = 5; // Use more in practice.
	 String featureSubsetStrategy = "auto"; // Let the algorithm choose.
	 String impurity1 = "gini";
	 Integer maxDepth1 = 5;
	 Integer maxBins1 = 32;
	 Integer seed1 = 12345;
	
	 RandomForestModel modelRF = RandomForest.trainClassifier(parsedData1, numberClasses1,categoricalFeaturesInfo1, numTrees, featureSubsetStrategy, impurity1, maxDepth1, maxBins1,seed1);
	
	 // Evaluate model on test instances and compute test error
	 JavaPairRDD<Double, Double> predictionAndLabelRF =
	   parsedData2.mapToPair(p -> new Tuple2<>(modelRF.predict(p.features()), p.label()));
	 double testErr =
	   predictionAndLabelRF.filter(pl -> !pl._1().equals(pl._2())).count() / (double) parsedData2.count();
	 System.out.println("Test Error: " + testErr);
	 System.out.println("Learned classification forest model:\n" + modelRF.toDebugString());**/
	 
	 /**NAIVES BAYES
	 NaiveBayesModel modelNB = NaiveBayes.train(parsedData1.rdd(), 1.0);
	 JavaPairRDD<Double, Double> predictionAndLabelNB =
	   parsedData2.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
	 double accuracyNB =
	   predictionAndLabelNB.filter(pl -> pl._1().equals(pl._2())).count() / (double) parsedData2.count();
	 System.out.println("Accuracy NB: " + accuracyNB);
	 
	 parsedData1.take(10).forEach(System.out::println);**/
	
	// ---------------------------------------- Save Model ---------------------------------------- //
			/**model.save(sc.sc(), "C://spark/spark-2.4.7-bin-hadoop2.7/data/mllib/LogisticRegressionModel");**/
			model.save(sc.sc(), "s3://dsqualitywine/LogisticRegressionModel/");
			
		// ---------------------------------------- Stop Spark Context ---------------------------------- //
    
    sc.stop();
     
  }
}
