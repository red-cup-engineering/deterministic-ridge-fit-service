import assert from "node:assert/strict";
import test from "node:test";
import { fitDeterministicRidge, predictDeterministicRidge } from "@red-cup-engineering/deterministic-ridge-fit-service";

const samples = [-1, 0, 1].map(x => ({ features: [x, 4], target: 3 + 2 * x }));

test("regularizes standardized features and leaves the intercept unpenalized", () => {
  const model = fitDeterministicRidge({ samples, lambda: "2" });
  assert.deepEqual(model.centers, [0, 4]);
  assert.deepEqual(model.scales, [1, 1]);
  assert.deepEqual(model.coefficients, [3, 1, 0]);
  assert.equal(predictDeterministicRidge(model, [2, "4"]), 5);
});

test("refuses missing training values and sparse prediction features", () => {
  assert.throws(() => fitDeterministicRidge({ samples: [samples[0], { features: [1, 4], target: null }] }),
    /ridge-sample-invalid/u);
  const model = fitDeterministicRidge({ samples });
  assert.throws(() => predictDeterministicRidge(model, [1, ,]), /ridge-prediction-input-invalid/u);
});

test("refuses moment overflow instead of returning a nonfinite model", () => {
  assert.throws(() => fitDeterministicRidge({
    samples: [{ features: [1e308], target: 1 }, { features: [1e308], target: 2 }],
  }), /ridge-nonfinite-moment/u);
});
