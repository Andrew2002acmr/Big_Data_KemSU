package lab5;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.fasterxml.jackson.databind.ObjectMapper;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

final class HospitalModel implements AutoCloseable {
    private final OrtEnvironment runtime = OrtEnvironment.getEnvironment();
    private final OrtSession model;
    private final JedisPool pool;
    private final ObjectMapper json = new ObjectMapper();
    private final Map<Long, float[]> cache = new LinkedHashMap<>(256, .75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<Long, float[]> eldest) {
            return size() > 4096;
        }
    };

    HospitalModel(String path, String redis) throws OrtException {
        try (var options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(1);
            options.setInterOpNumThreads(1);
            model = runtime.createSession(path, options);
        }
        pool = new JedisPool(redis, 6379);
    }

    float[] features(StreamJob.Event event) throws IOException {
        float[] profile = event.cached ? cache.get(event.patient) : null;
        if (profile == null) {
            try (Jedis redis = pool.getResource()) {
                String raw = redis.get("lab5:feature:v1:" + event.patient);
                if (raw == null) {
                    return null;
                }
                profile = json.readValue(raw, float[].class);
            }
            if (profile.length != 4) {
                throw new IOException("Invalid feature snapshot size");
            }
            if (event.cached) {
                cache.put(event.patient, profile);
            }
        }
        return new float[] {profile[0], profile[1], profile[2], profile[3], event.stay,
                "Rehab".equals(event.destination) ? 1f : 0f,
                "Nursing_Facility".equals(event.destination) ? 1f : 0f};
    }

    float predict(float[] features) throws OrtException {
        try (OnnxTensor tensor = OnnxTensor.createTensor(runtime, new float[][] {features});
             OrtSession.Result result = model.run(Map.of("input", tensor))) {
            return ((float[][]) result.get(0).getValue())[0][0];
        }
    }

    @Override
    public void close() throws OrtException {
        pool.close();
        model.close();
    }
}
