import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.ClassificationScoreCalculator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.scorecalc.RegressionScoreCalculator;
import org.deeplearning4j.earlystopping.termination.InvalidScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.AdamUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.ui.stats.StatsListener;
import scala.Equals;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.TimeUnit;


public class TrainBest {
    public static void main(String[] args) throws Exception {
        Properties prop = new Properties();
        InputStream input = null;

        try {
            input = new FileInputStream("/home/iva/progs/unsupervised-ad/src/main/resources/ad.properties");

            // load a properties file
            prop.load(input);
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            if (input != null) {
                try {
                    input.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        final String filenameTrain = prop.getProperty("filenameTrain");
        final String filenameTest = prop.getProperty("filenameTest");
        final String filenameSerTo = prop.getProperty("filenameSerTo");
        int seed = Integer.parseInt(prop.getProperty("seed"));
        double learningRate = Double.parseDouble(prop.getProperty("learningRate"));
        int batchSize = Integer.parseInt(prop.getProperty("batchSize"));
        int batchSize1 = Integer.parseInt(prop.getProperty("batchSize1"));
        int nEpochs = Integer.parseInt(prop.getProperty("numEpochs1"));
        int numInputs = Integer.parseInt(prop.getProperty("numInputs"));
        int numOutputs = Integer.parseInt(prop.getProperty("numOutputs"));
        int numHiddenNodes = Integer.parseInt(prop.getProperty("numHiddenNodes"));
        int labelIndex = Integer.parseInt(prop.getProperty("labelIndex"));
        int numPosLabels = Integer.parseInt(prop.getProperty("numPosLabels"));


        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numPosLabels);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize1, labelIndex, numPosLabels);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .biasInit(1.0)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(0.19132168863349655)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.SQUARED_LOSS).nIn(numHiddenNodes).nOut(numOutputs)
                        .build())
                .pretrain(false).backprop(true).build();
        //MultiLayerNetwork model = new MultiLayerNetwork(conf);
        //model.init();

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new LocalFileModelSaver(filenameSerTo);
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(nEpochs))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(480, TimeUnit.MINUTES))
                .scoreCalculator(new ClassificationScoreCalculator(Evaluation.Metric.F1, testIter))
                .iterationTerminationConditions(new InvalidScoreIterationTerminationCondition())
                .evaluateEveryNEpochs(1)
                //.modelSaver(saver)
                //.saveLastModel(true)
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf,trainIter);
        EarlyStoppingResult result = trainer.fit();

        //Print out the results:
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList<Integer>(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Score vs. Epoch:");
        for( Integer i : list){
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }
    }
}
