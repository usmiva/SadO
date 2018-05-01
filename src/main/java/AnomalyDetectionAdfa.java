import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.listener.DL4JArbiterStatusReportingListener;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.saving.InMemoryResultSaver;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.scoring.impl.TestSetF1ScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.arbiter.ui.listener.ArbiterStatusListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

public class AnomalyDetectionAdfa {
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
        int batchSize = Integer.parseInt(prop.getProperty("batchSize"));
        int batchSize1 = Integer.parseInt(prop.getProperty("batchSize1"));
        int numEpochs = Integer.parseInt(prop.getProperty("numEpochs"));
        int numInputs = Integer.parseInt(prop.getProperty("numInputs"));
        int numOutputs = Integer.parseInt(prop.getProperty("numOutputs"));
        int labelIndex = Integer.parseInt(prop.getProperty("labelIndex"));
        int labelIndexFrom = Integer.parseInt(prop.getProperty("labelIndexFrom"));
        int labelIndexTo = Integer.parseInt(prop.getProperty("labelIndexTo"));
        int numPosLabels = Integer.parseInt(prop.getProperty("numPosLabels"));

        ParameterSpace<Integer> layerSize = new IntegerParameterSpace(3, 180);
        ParameterSpace<Integer> layerSize1 = new IntegerParameterSpace(10, 50);
        ParameterSpace<Integer> layerSize2 = new IntegerParameterSpace(150, 170);
        ParameterSpace<Double> learningRate = new ContinuousParameterSpace(0.0001, 1.0);
        ParameterSpace<Double> gradientTreshold = new ContinuousParameterSpace(-1, 1);
        ParameterSpace<Double> beta1 = new ContinuousParameterSpace(0.9, 1.0);
        ParameterSpace<Double> beta2 = new ContinuousParameterSpace(0.9, 1.0);
        ParameterSpace<Double> epsilon = new ContinuousParameterSpace(10E-10, 10E-6);
        ParameterSpace<Double> e = new ContinuousParameterSpace(-1, 1);

        MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
                .updater(new AdamSpace(learningRate, beta1, beta2, epsilon))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.trainingWorkspaceMode(WorkspaceMode.SINGLE)
                .weightInit(WeightInit.UNIFORM)
                .activation(Activation.SOFTMAX)
                .biasInit(1.0)
                //.l2(e)
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                //.gradientNormalizationThreshold(gradientTreshold)
                .addLayer(new DenseLayerSpace.Builder()
                        .nIn(numInputs).nOut(layerSize).build())
                .addLayer(new OutputLayerSpace.Builder()
                        .nIn(layerSize).lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).nOut(numOutputs)
                        .build())
                .pretrain(false).backprop(true).build();
        CandidateGenerator generator = new RandomSearchGenerator(hyperparameterSpace, null);


        DataProvider dataProvider = new CSVDataProvider(numEpochs, batchSize, batchSize1, filenameTrain, filenameTest, labelIndex, numPosLabels);
        String baseSaveDirectory = "anomaly-testing1/";
        File f = new File(baseSaveDirectory);
        if (f.exists()) f.delete();
        f.mkdir();
        ResultSaver saver = new FileModelSaver(baseSaveDirectory);
        ScoreFunction sf = new EvaluationScoreFunction(Evaluation.Metric.ACCURACY);
        TerminationCondition[] terminationConditions = {new MaxTimeCondition(600, TimeUnit.MINUTES), new MaxCandidatesCondition(500)};

        OptimizationConfiguration conf = new OptimizationConfiguration.Builder()
                .candidateGenerator(generator)
                .dataProvider(dataProvider)
                .modelSaver(saver)
                .scoreFunction(sf)
                .terminationConditions(terminationConditions).build();
        IOptimizationRunner run = new LocalOptimizationRunner(conf, new MultiLayerNetworkTaskCreator());
        StatsStorage ss = new FileStatsStorage(new File("UiStats.dl4j"));
        run.addListeners(new ArbiterStatusListener(ss));
        UIServer.getInstance().attach(ss);
        run.execute();

        StringBuilder sb = new StringBuilder();
        sb.append("Best score: ").append(run.bestScore()).append("\n")
                .append("Index of model with best score: ").append(run.bestScoreCandidateIndex()).append("\n")
                .append("Number of configurations evaluated: ").append(run.numCandidatesCompleted()).append("\n");
        System.out.println(sb.toString());

        int indexOfBestResult = run.bestScoreCandidateIndex();
        List<ResultReference> allResults = run.getResults();
        OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
        MultiLayerNetwork bestModel = (MultiLayerNetwork) bestResult.getResult();

        System.out.println("\n\nConfiguration of best model:\n");
        System.out.println(bestModel.getLayerWiseConfigurations().toString());
        //Thread.sleep(6000);
        UIServer.getInstance().stop();
    }
    public static class CSVDataProvider implements DataProvider {
        private int numEpochs;
        private int batchSize;
        private int batchSize1;
        private int labelIndex;
        private int labelIndexFrom;
        private int labelIndexTo;
        private int numPosLabels;
        private String filenameTrain;
        private String filenameTest;

        public CSVDataProvider(@JsonProperty("numEpochs") int numEpochs, @JsonProperty("batchSize") int batchSize, @JsonProperty("batchSize1") int batchSize1, @JsonProperty("filenameTrain") String filenameTrain, @JsonProperty("filenameTest") String filenameTest, @JsonProperty("labelIndex") int labelIndex, @JsonProperty("numPosLabels") int numPosLabels) {
            this.numEpochs = numEpochs;
            this.batchSize = batchSize;
            this.batchSize1 = batchSize1;
            this.filenameTrain = filenameTrain;
            this.filenameTest = filenameTest;
            this.labelIndex = labelIndex;
            this.labelIndexFrom = labelIndexFrom;
            this.labelIndexTo = labelIndexTo;
            this.numPosLabels = numPosLabels;
        }
        public DataSetIterator trainData(Map<String, Object> dataParameters) {
            try {
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(new File(filenameTrain)));

                return new MultipleEpochsIterator(numEpochs, new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numPosLabels));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        public DataSetIterator testData(Map<String, Object> dataParameters) {
            try {
                RecordReader rrTest = new CSVRecordReader();
                rrTest.initialize(new FileSplit(new File(filenameTest)));
                return new RecordReaderDataSetIterator(rrTest, batchSize1, labelIndex, numPosLabels);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        public Class<?> getDataType() { return DataSetIterator.class; }
    }
}

