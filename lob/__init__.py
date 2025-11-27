"""
L3 Limit Order Book (LOB) module for equity market simulation.

This module provides high-fidelity order book simulation for US equities,
supporting full market microstructure modeling including:
- Price-time priority (FIFO) queue tracking
- FIFO matching engine with self-trade prevention
- Queue position tracking (MBO/MBP estimation)
- Order lifecycle management
- Iceberg/hidden order simulation
- Market-by-Order (MBO) and Market-by-Price (MBP) views
- LOBSTER message format parsing
- Fill probability models (Poisson, Queue-Reactive, Historical)
- Queue value computation (Moallemi & Yuan)
- Model calibration from historical data
- Market impact models (Kyle, Almgren-Chriss, Gatheral)
- Impact effects on LOB state (quote shifting, liquidity reaction)
- Impact calibration from historical trade data
- Latency simulation (feed/order/exchange/fill latencies)
- Event scheduling with priority queue and race condition detection

Architecture:
    This module is SEPARATE from the Cython LOB implementations (fast_lob.pyx,
    execlob_book.pyx) which are optimized for crypto environments. The Python
    implementation here provides better flexibility and testability for equity
    market simulation while integrating with execution_providers.py.

Stage 1 (v1.0): Data structures, parsers, state manager
Stage 2 (v2.0): Matching engine, queue tracker, order manager
Stage 3 (v3.0): Fill probability models, queue value, calibration
Stage 4 (v4.0): Market impact models, effects, impact calibration
Stage 5 (v5.0): Latency simulation, event scheduler
Stage 6 (v6.0): Hidden liquidity detection, dark pool simulation
Stage 8 (v8.0): Data adapters, unified L3 calibration pipeline

Usage:
    from lob import OrderBook, LimitOrder, PriceLevel
    from lob.parsers import LOBSTERParser
    from lob.state_manager import LOBStateManager
    from lob.matching_engine import MatchingEngine
    from lob.queue_tracker import QueuePositionTracker
    from lob.order_manager import OrderManager
    from lob.fill_probability import (
        FillProbabilityModel, QueueReactiveModel, create_fill_probability_model
    )
    from lob.queue_value import QueueValueModel, create_queue_value_model
    from lob.calibration import CalibrationPipeline, create_calibration_pipeline
    from lob.market_impact import (
        MarketImpactModel, AlmgrenChrissModel, GatheralModel, create_impact_model
    )
    from lob.impact_effects import ImpactEffects, LOBImpactSimulator
    from lob.impact_calibration import ImpactCalibrationPipeline
    from lob.latency_model import (
        LatencyModel, LatencyProfile, create_latency_model
    )
    from lob.event_scheduler import (
        EventScheduler, SimulationClock, create_event_scheduler
    )
    from lob.hidden_liquidity import (
        IcebergDetector, IcebergOrder, HiddenLiquidityEstimator,
        create_iceberg_detector, create_hidden_liquidity_estimator
    )
    from lob.dark_pool import (
        DarkPoolSimulator, DarkPoolVenue, DarkPoolFill,
        create_dark_pool_simulator, create_default_dark_pool_simulator
    )

Note:
    This module does NOT affect crypto execution paths. Crypto continues
    to use the existing Cython LOB implementations.
"""

from lob.data_structures import (
    Side,
    OrderType,
    LimitOrder,
    PriceLevel,
    OrderBook,
    Fill,
    Trade,
)

from lob.state_manager import (
    LOBStateManager,
    LOBSnapshot,
    LOBMessage,
    MessageType,
)

from lob.parsers import (
    LOBSTERParser,
    LOBSTERMessage,
)

from lob.matching_engine import (
    MatchingEngine,
    MatchResult,
    MatchType,
    STPAction,
    ProRataMatchingEngine,
    create_matching_engine,
)

from lob.queue_tracker import (
    QueuePositionTracker,
    QueueState,
    FillProbability,
    PositionEstimationMethod,
    LevelStatistics,
    create_queue_tracker,
)

from lob.order_manager import (
    OrderManager,
    ManagedOrder,
    OrderEvent,
    OrderEventType,
    OrderLifecycleState,
    TimeInForce,
    create_order_manager,
)

# Stage 3: Fill Probability & Queue Value (v3.0)
from lob.fill_probability import (
    FillProbabilityModel,
    FillProbabilityModelType,
    FillProbabilityResult,
    LOBState,
    AnalyticalPoissonModel,
    QueueReactiveModel,
    HistoricalRateModel,
    HistoricalFillRate,
    DistanceBasedModel,
    CompositeFillProbabilityModel,
    create_fill_probability_model,
    compute_fill_probability_for_order,
)

from lob.queue_value import (
    QueueValueModel,
    QueueValueResult,
    QueueValueConfig,
    QueueValueTracker,
    OrderDecision,
    AdverseSelectionParams,
    DynamicQueueValueModel,
    SpreadDecompositionModel,
    QueueImprovementEstimator,
    create_queue_value_model,
    compute_order_queue_value,
)

from lob.calibration import (
    CalibrationPipeline,
    CalibrationResult,
    CrossValidationResult,
    TradeRecord,
    OrderRecord,
    PoissonRateCalibrator,
    QueueReactiveCalibrator,
    AdverseSelectionCalibrator,
    HistoricalRateCalibrator,
    create_calibration_pipeline,
    calibrate_from_trades,
    estimate_arrival_rate,
)

# Stage 4: Market Impact Models (v4.0)
from lob.market_impact import (
    MarketImpactModel,
    ImpactModelType,
    ImpactParameters,
    ImpactResult,
    ImpactState,
    DecayType,
    KyleLambdaModel,
    AlmgrenChrissModel,
    GatheralModel,
    CompositeImpactModel,
    ImpactTracker,
    create_impact_model,
    create_impact_tracker,
)

from lob.impact_effects import (
    ImpactEffects,
    ImpactEffectConfig,
    LOBImpactSimulator,
    QuoteShiftType,
    QuoteShiftResult,
    LiquidityReaction,
    LiquidityReactionResult,
    MomentumSignal,
    MomentumResult,
    AdverseSelectionResult,
    create_impact_effects,
    create_lob_impact_simulator,
)

from lob.impact_calibration import (
    ImpactCalibrationPipeline,
    TradeObservation,
    CalibrationDataset,
    CalibrationResult as ImpactCalibrationResult,
    CrossValidationResult as ImpactCrossValidationResult,
    AlmgrenChrissCalibrator,
    GatheralDecayCalibrator,
    KyleLambdaCalibrator,
    RollingImpactCalibrator,
    create_calibrator,
    create_calibration_pipeline as create_impact_calibration_pipeline,
    calibrate_from_trades as calibrate_impact_from_trades,
)

# Stage 5: Latency Simulation (v5.0)
from lob.latency_model import (
    LatencyConfig,
    LatencyDistribution,
    LatencyModel,
    LatencyModelConfig,
    LatencyProfile,
    LatencySample,
    LatencySampler,
    create_latency_model,
)

from lob.event_scheduler import (
    EventScheduler,
    EventType,
    FillEvent,
    MarketDataEvent,
    OrderSubmission,
    RaceConditionInfo,
    ScheduledEvent,
    SimulationClock,
    create_event_scheduler,
    create_simulation_clock,
)

# Stage 6: Hidden Liquidity & Dark Pools (v6.0)
from lob.hidden_liquidity import (
    IcebergDetector,
    IcebergOrder,
    IcebergState,
    RefillEvent,
    LevelSnapshot,
    ExecutionEvent,
    DetectionConfidence,
    HiddenLiquidityEstimator,
    create_iceberg_detector,
    create_hidden_liquidity_estimator,
)

from lob.dark_pool import (
    DarkPoolSimulator,
    DarkPoolVenue,
    DarkPoolConfig,
    DarkPoolFill,
    DarkPoolState,
    DarkPoolVenueType,
    FillType,
    LeakageType,
    InformationLeakage,
    create_dark_pool_simulator,
    create_default_dark_pool_simulator,
)

# Stage 7: L3 Execution Provider Configuration (v7.0)
from lob.config import (
    # Enums
    LatencyDistributionType,
    LatencyProfileType,
    FillProbabilityModelType as L3FillProbabilityModelType,
    ImpactModelType as L3ImpactModelType,
    DecayFunctionType,
    QueueEstimationMethod,
    LOBDataSourceType,
    # Configuration Classes
    LatencyComponentConfig,
    LatencyConfig as L3LatencyConfig,
    FillProbabilityConfig,
    QueueValueConfig as L3QueueValueConfig,
    MarketImpactConfig,
    IcebergConfig,
    HiddenLiquidityConfig,
    DarkPoolVenueConfig,
    DarkPoolsConfig,
    QueueTrackingConfig,
    EventSchedulingConfig,
    LOBDataConfig,
    L3ExecutionConfig,
    # Factory Functions
    create_l3_config,
    # Backward Compatibility
    latency_config_to_dataclass,
    impact_config_to_parameters,
)

# Stage 8: Data Adapters & Unified Calibration Pipeline (v8.0)
from lob.data_adapters import (
    # Base adapter
    BaseLOBAdapter,
    # Format-specific adapters
    LOBSTERAdapter,
    ITCHAdapter,
    BinanceL2Adapter,
    AlpacaL2Adapter,
    # Data structures
    LOBUpdate,
    DepthLevel,
    DepthSnapshot,
    AdapterStats,
    DataSourceType,
    # Factory functions
    create_lob_adapter,
    load_orderbook_from_file,
    get_supported_formats,
)

from lob.calibration_pipeline import (
    # Main calibration pipeline
    L3CalibrationPipeline,
    # Component calibrators
    LatencyCalibrator,
    QueueDynamicsCalibrator,
    # Data structures
    LatencyObservation,
    QuoteObservation,
    LatencyDistributionParams,
    LatencyCalibrationResult,
    QueueDynamicsResult,
    L3CalibrationResult,
    CalibrationDataType,
    # Factory functions
    create_calibration_pipeline as create_l3_calibration_pipeline,
    calibrate_from_dataframe,
)

__all__ = [
    # Core data structures
    "Side",
    "OrderType",
    "LimitOrder",
    "PriceLevel",
    "OrderBook",
    "Fill",
    "Trade",
    # State management
    "LOBStateManager",
    "LOBSnapshot",
    "LOBMessage",
    "MessageType",
    # Parsers
    "LOBSTERParser",
    "LOBSTERMessage",
    # Matching engine (Stage 2)
    "MatchingEngine",
    "MatchResult",
    "MatchType",
    "STPAction",
    "ProRataMatchingEngine",
    "create_matching_engine",
    # Queue tracking (Stage 2)
    "QueuePositionTracker",
    "QueueState",
    "FillProbability",
    "PositionEstimationMethod",
    "LevelStatistics",
    "create_queue_tracker",
    # Order management (Stage 2)
    "OrderManager",
    "ManagedOrder",
    "OrderEvent",
    "OrderEventType",
    "OrderLifecycleState",
    "TimeInForce",
    "create_order_manager",
    # Fill Probability Models (Stage 3)
    "FillProbabilityModel",
    "FillProbabilityModelType",
    "FillProbabilityResult",
    "LOBState",
    "AnalyticalPoissonModel",
    "QueueReactiveModel",
    "HistoricalRateModel",
    "HistoricalFillRate",
    "DistanceBasedModel",
    "CompositeFillProbabilityModel",
    "create_fill_probability_model",
    "compute_fill_probability_for_order",
    # Queue Value (Stage 3)
    "QueueValueModel",
    "QueueValueResult",
    "QueueValueConfig",
    "QueueValueTracker",
    "OrderDecision",
    "AdverseSelectionParams",
    "DynamicQueueValueModel",
    "SpreadDecompositionModel",
    "QueueImprovementEstimator",
    "create_queue_value_model",
    "compute_order_queue_value",
    # Calibration (Stage 3)
    "CalibrationPipeline",
    "CalibrationResult",
    "CrossValidationResult",
    "TradeRecord",
    "OrderRecord",
    "PoissonRateCalibrator",
    "QueueReactiveCalibrator",
    "AdverseSelectionCalibrator",
    "HistoricalRateCalibrator",
    "create_calibration_pipeline",
    "calibrate_from_trades",
    "estimate_arrival_rate",
    # Market Impact Models (Stage 4)
    "MarketImpactModel",
    "ImpactModelType",
    "ImpactParameters",
    "ImpactResult",
    "ImpactState",
    "DecayType",
    "KyleLambdaModel",
    "AlmgrenChrissModel",
    "GatheralModel",
    "CompositeImpactModel",
    "ImpactTracker",
    "create_impact_model",
    "create_impact_tracker",
    # Impact Effects (Stage 4)
    "ImpactEffects",
    "ImpactEffectConfig",
    "LOBImpactSimulator",
    "QuoteShiftType",
    "QuoteShiftResult",
    "LiquidityReaction",
    "LiquidityReactionResult",
    "MomentumSignal",
    "MomentumResult",
    "AdverseSelectionResult",
    "create_impact_effects",
    "create_lob_impact_simulator",
    # Impact Calibration (Stage 4)
    "ImpactCalibrationPipeline",
    "TradeObservation",
    "CalibrationDataset",
    "ImpactCalibrationResult",
    "ImpactCrossValidationResult",
    "AlmgrenChrissCalibrator",
    "GatheralDecayCalibrator",
    "KyleLambdaCalibrator",
    "RollingImpactCalibrator",
    "create_calibrator",
    "create_impact_calibration_pipeline",
    "calibrate_impact_from_trades",
    # Latency Simulation (Stage 5)
    "LatencyConfig",
    "LatencyDistribution",
    "LatencyModel",
    "LatencyModelConfig",
    "LatencyProfile",
    "LatencySample",
    "LatencySampler",
    "create_latency_model",
    # Event Scheduler (Stage 5)
    "EventScheduler",
    "EventType",
    "FillEvent",
    "MarketDataEvent",
    "OrderSubmission",
    "RaceConditionInfo",
    "ScheduledEvent",
    "SimulationClock",
    "create_event_scheduler",
    "create_simulation_clock",
    # Hidden Liquidity (Stage 6)
    "IcebergDetector",
    "IcebergOrder",
    "IcebergState",
    "RefillEvent",
    "LevelSnapshot",
    "ExecutionEvent",
    "DetectionConfidence",
    "HiddenLiquidityEstimator",
    "create_iceberg_detector",
    "create_hidden_liquidity_estimator",
    # Dark Pool Simulation (Stage 6)
    "DarkPoolSimulator",
    "DarkPoolVenue",
    "DarkPoolConfig",
    "DarkPoolFill",
    "DarkPoolState",
    "DarkPoolVenueType",
    "FillType",
    "LeakageType",
    "InformationLeakage",
    "create_dark_pool_simulator",
    "create_default_dark_pool_simulator",
    # L3 Configuration (Stage 7)
    "LatencyDistributionType",
    "LatencyProfileType",
    "L3FillProbabilityModelType",
    "L3ImpactModelType",
    "DecayFunctionType",
    "QueueEstimationMethod",
    "LOBDataSourceType",
    "LatencyComponentConfig",
    "L3LatencyConfig",
    "FillProbabilityConfig",
    "L3QueueValueConfig",
    "MarketImpactConfig",
    "IcebergConfig",
    "HiddenLiquidityConfig",
    "DarkPoolVenueConfig",
    "DarkPoolsConfig",
    "QueueTrackingConfig",
    "EventSchedulingConfig",
    "LOBDataConfig",
    "L3ExecutionConfig",
    "create_l3_config",
    "latency_config_to_dataclass",
    "impact_config_to_parameters",
    # Data Adapters (Stage 8)
    "BaseLOBAdapter",
    "LOBSTERAdapter",
    "ITCHAdapter",
    "BinanceL2Adapter",
    "AlpacaL2Adapter",
    "LOBUpdate",
    "DepthLevel",
    "DepthSnapshot",
    "AdapterStats",
    "DataSourceType",
    "create_lob_adapter",
    "load_orderbook_from_file",
    "get_supported_formats",
    # Unified Calibration Pipeline (Stage 8)
    "L3CalibrationPipeline",
    "LatencyCalibrator",
    "QueueDynamicsCalibrator",
    "LatencyObservation",
    "QuoteObservation",
    "LatencyDistributionParams",
    "LatencyCalibrationResult",
    "QueueDynamicsResult",
    "L3CalibrationResult",
    "CalibrationDataType",
    "create_l3_calibration_pipeline",
    "calibrate_from_dataframe",
]

__version__ = "8.0.0"
