#include <catch.hpp>

#include <memory>
#include <string>
#include <vector>

#include <Api/KttException.h>
#include <Api/Objective/ObjectiveFactory.h>
#include <Api/Objective/ObjectiveType.h>
#include <Api/Objective/TuningObjective.h>
#include <Api/Output/ComputationResult.h>
#include <Api/Output/KernelResult.h>
#include <Api/Configuration/DimensionVector.h>
#include <Api/Configuration/KernelConfiguration.h>
#include <Tuner.h>

namespace
{

// Builds a valid kernel result with the given total duration (ns) and energy (J)
// carried directly as energy data, mimicking an NVML-backed measurement.
ktt::KernelResult MakeResultWithEnergy(const ktt::Nanoseconds duration, const double energyJoules)
{
    ktt::ComputationResult computation("vectorAddition");
    computation.SetDurationData(duration, 0, 0);
    computation.SetSizeData(ktt::DimensionVector(1024), ktt::DimensionVector(32));
    computation.SetEnergyConsumption(energyJoules);

    ktt::KernelConfiguration configuration;
    return ktt::KernelResult("kernel", configuration, {computation}, "timestamp");
}

// Builds a valid kernel result with power-only data (mW). Energy is derived as
// power * duration, mimicking a Sysman-style power sampling measurement.
ktt::KernelResult MakeResultWithPower(const ktt::Nanoseconds duration, const uint32_t powerMilliwatts)
{
    ktt::ComputationResult computation("vectorAddition");
    computation.SetDurationData(duration, 0, 0);
    computation.SetSizeData(ktt::DimensionVector(1024), ktt::DimensionVector(32));
    computation.SetPowerUsage(powerMilliwatts);

    ktt::KernelConfiguration configuration;
    return ktt::KernelResult("kernel", configuration, {computation}, "timestamp");
}

} // namespace

TEST_CASE("Weighted energy-performance objective validates its inputs", "[Objective]")
{
    SECTION("Negative or zero-sum weights are rejected")
    {
        REQUIRE_THROWS_AS(ktt::WeightedEnergyPerformanceObjective(-0.5, 0.5, 1000.0, 1.0), ktt::KttException);
        REQUIRE_THROWS_AS(ktt::WeightedEnergyPerformanceObjective(0.0, 0.0, 1000.0, 1.0), ktt::KttException);
    }

    SECTION("Non-positive references are rejected")
    {
        REQUIRE_THROWS_AS(ktt::WeightedEnergyPerformanceObjective(0.5, 0.5, 0.0, 1.0), ktt::KttException);
        REQUIRE_THROWS_AS(ktt::WeightedEnergyPerformanceObjective(0.5, 0.5, 1000.0, -1.0), ktt::KttException);
    }

    SECTION("Weights are normalized to sum to one")
    {
        ktt::WeightedEnergyPerformanceObjective objective(1.0, 3.0, 1000.0, 1.0);
        REQUIRE(objective.GetPerformanceWeight() == Approx(0.25));
        REQUIRE(objective.GetEnergyWeight() == Approx(0.75));
    }
}

TEST_CASE("Weighted energy-performance objective scores results", "[Objective]")
{
    // Reference: 1000 ns, 2.0 J. Result under test: 500 ns, 1.0 J.
    // Expected score with 0.5/0.5 weights: 0.5 * 0.5 + 0.5 * 0.5 = 0.5.
    ktt::WeightedEnergyPerformanceObjective objective(0.5, 0.5, 1000.0, 2.0);
    const auto result = MakeResultWithEnergy(500, 1.0);

    SECTION("Score combines normalized duration and energy")
    {
        REQUIRE(objective.CanEvaluate(result));
        REQUIRE(objective.Evaluate(result) == Approx(0.5));
        REQUIRE(objective.GetName() == "WeightedEnergyPerformance");
    }

    SECTION("Metadata carries weights and references")
    {
        const auto metadata = objective.CreateMetadata(0.5);
        REQUIRE(metadata.name == "WeightedEnergyPerformance");
        REQUIRE(metadata.score == Approx(0.5));
        REQUIRE(metadata.performanceWeight.value() == Approx(0.5));
        REQUIRE(metadata.energyWeight.value() == Approx(0.5));
        REQUIRE(metadata.referenceDuration.value() == Approx(1000.0));
        REQUIRE(metadata.referenceEnergy.value() == Approx(2.0));
    }

    SECTION("Results without energy or power data cannot be evaluated")
    {
        ktt::ComputationResult bare("vectorAddition");
        bare.SetDurationData(500, 0, 0);

        ktt::KernelConfiguration configuration;
        const ktt::KernelResult bareResult("kernel", configuration, {bare}, "timestamp");

        REQUIRE(!objective.CanEvaluate(bareResult));
        REQUIRE_THROWS_AS(objective.Evaluate(bareResult), ktt::KttException);
    }

    SECTION("Power-only results are evaluated through derived energy")
    {
        // 1 W over 1 s yields 1 J: 1000 mW over 1e9 ns.
        const auto powerResult = MakeResultWithPower(1'000'000'000, 1000);
        REQUIRE(objective.CanEvaluate(powerResult));

        ktt::WeightedEnergyPerformanceObjective unitObjective(0.0, 1.0, 1.0, 1.0);
        REQUIRE(unitObjective.Evaluate(powerResult) == Approx(1.0));
    }
}

TEST_CASE("Duration objective and objective factory", "[Objective]")
{
    SECTION("Duration objective evaluates to total duration")
    {
        ktt::DurationObjective objective;
        const auto result = MakeResultWithEnergy(1234, 1.0);

        REQUIRE(objective.CanEvaluate(result));
        REQUIRE(objective.Evaluate(result) == Approx(1234.0));
        REQUIRE(objective.GetName() == "Duration");
    }

    SECTION("Factory creates both supported objective types")
    {
        ktt::ObjectiveSpec durationSpec;
        durationSpec.type = ktt::ObjectiveType::Duration;
        REQUIRE(CreateObjective(durationSpec)->GetName() == "Duration");

        ktt::ObjectiveSpec weightedSpec;
        weightedSpec.type = ktt::ObjectiveType::WeightedEnergyPerformance;
        weightedSpec.performanceWeight = 0.5;
        weightedSpec.energyWeight = 0.5;
        weightedSpec.referenceDuration = 1000.0;
        weightedSpec.referenceEnergy = 1.0;
        REQUIRE(CreateObjective(weightedSpec)->GetName() == "WeightedEnergyPerformance");
    }
}

#ifdef KTT_API_CUDA
TEST_CASE("Energy objective end-to-end on CUDA device", "[Objective]")
{
    // GPU-gated: self-skip cleanly when no CUDA device or driver is present.
    std::unique_ptr<ktt::Tuner> tuner;

    try
    {
        tuner = std::make_unique<ktt::Tuner>(0, 0, ktt::ComputeApi::CUDA);
    }
    catch (const ktt::KttException& exception)
    {
        WARN("Skipping CUDA objective test, no usable CUDA device: " + std::string(exception.what()));
        return;
    }

    const std::string source =
        "__global__ void vectorAddition(const float* a, const float* b, float* result, const float scalar)\n"
        "{\n"
        "    const int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    result[index] = a[index] + b[index] + scalar;\n"
        "}\n";

    const size_t numberOfElements = 1024;
    std::vector<float> a(numberOfElements, 1.0f);
    std::vector<float> b(numberOfElements, 2.0f);
    std::vector<float> result(numberOfElements, 0.0f);
    const float scalarValue = 3.0f;

    const ktt::KernelDefinitionId definition = tuner->AddKernelDefinition("vectorAddition", source,
        ktt::DimensionVector(numberOfElements), ktt::DimensionVector(32), {"float"});
    const ktt::ArgumentId aId = tuner->AddArgumentVector(a, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId bId = tuner->AddArgumentVector(b, ktt::ArgumentAccessType::ReadOnly);
    const ktt::ArgumentId resultId = tuner->AddArgumentVector(result, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId scalarId = tuner->AddArgumentScalar(scalarValue);
    tuner->SetArguments(definition, {aId, bId, resultId, scalarId});

    const ktt::KernelId kernel = tuner->CreateSimpleKernel("Addition", definition);

    // At least one tuning parameter is required, otherwise no configurations
    // are generated and Tune() returns an empty result set.
    tuner->AddParameter(kernel, "BLOCK_SIZE", std::vector<uint64_t>{32, 64});

    tuner->SetTuningObjective(kernel,
        std::make_unique<ktt::WeightedEnergyPerformanceObjective>(0.5, 0.5, 1'000'000.0, 0.01));

    const std::vector<ktt::KernelResult> results = tuner->Tune(kernel);

    REQUIRE(!results.empty());
    REQUIRE(results.front().IsValid());

    // Energy data is only present when the power manager initialized (NVML on
    // NVIDIA builds with --power-usage). The objective must accept the result
    // exactly when energy data is available.
    const bool hasEnergy = !results.front().GetResults().empty()
        && (results.front().GetResults().front().HasEnergyData()
            || results.front().GetResults().front().HasPowerData());

    ktt::WeightedEnergyPerformanceObjective objective(0.5, 0.5, 1'000'000.0, 0.01);
    REQUIRE(objective.CanEvaluate(results.front()) == hasEnergy);
}
#endif // KTT_API_CUDA
