# PEI E-commerce Test Suite

This directory contains comprehensive test cases for the PEI E-commerce data processing pipeline. The test suite has been significantly enhanced to provide thorough coverage of all modules and edge cases.

## Test Structure

### Original Test Files
- `test_enriched_order_details.py` - Basic tests for order details enrichment
- `test_agg_profit.py` - Basic tests for profit aggregation
- `test_enriched_customer_products.py` - Basic tests for customer/product enrichment

### Extended Test Files (New)
- `test_enriched_order_details_extended.py` - Comprehensive edge case testing for order details
- `test_agg_profit_extended.py` - Advanced aggregation and date handling tests
- `test_enriched_customer_products_extended.py` - Thorough customer/product validation tests
- `test_integration.py` - End-to-end pipeline integration tests

## Test Coverage Improvements

### Key Areas Enhanced

1. **Edge Case Testing**
   - Empty DataFrames
   - Null value handling
   - Boundary conditions (zero profits, edge dates)
   - Special characters in data
   - Large dataset simulation

2. **Schema Validation**
   - Missing table error handling
   - Missing column validation
   - Column name case sensitivity
   - Comprehensive schema checks

3. **Audit Column Testing**
   - Timestamp column presence
   - Job ID and Run ID validation
   - Audit trail verification

4. **Join Behavior Testing**
   - Left join scenarios
   - Missing reference data
   - Orphaned records handling

5. **Data Quality Validation**
   - Profit rounding verification
   - Date format validation
   - Negative value handling
   - Data type consistency

6. **Integration Testing**
   - End-to-end pipeline flow
   - Data lineage verification
   - Performance with large datasets
   - Pipeline idempotency
   - Multi-stage error handling

7. **Utility Function Testing**
   - `get_spark()` function validation
   - Session management testing

## Running Tests

### Prerequisites
```bash
# Install required dependencies
python -m pip install pytest pytest-cov pytest-html pyspark delta-spark
```

### Quick Start
```bash
# Run all tests
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run specific test suites
python run_tests.py --basic      # Original tests only
python run_tests.py --extended   # Extended tests only
python run_tests.py --integration # Integration tests only
```

### Advanced Usage
```bash
# Install dependencies automatically
python run_tests.py --install-deps

# Run tests in parallel (faster)
python run_tests.py --parallel

# Generate HTML report
python run_tests.py --html-report

# Run performance tests
python run_tests.py --performance
```

### Manual pytest Commands
```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_enriched_order_details_extended.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run integration tests only
pytest tests/test_integration.py -v -s
```

## Test Categories

### Unit Tests
- Individual function testing
- Input validation
- Error handling
- Data transformation logic

### Integration Tests
- End-to-end pipeline workflows
- Multi-module interactions
- Data flow validation
- Performance testing

### Extended Tests
- Edge cases and boundary conditions
- Large dataset handling
- Special character support
- Comprehensive error scenarios

## Test Data Patterns

### Valid Test Data
```python
# Orders with positive profits and valid references
Row(Order_ID="O1", Order_Date="21/8/2016", Customer_ID="C1", Product_ID="P1", Profit=100.0)

# Customers with all required fields
Row(Customer_ID="C1", Customer_Name="Alice", Country="USA", ...)

# Products with positive prices
Row(Product_ID="P1", Category="Tech", Sub_Category="Phones", Price_per_product=500.0)
```

### Invalid Test Data
```python
# Orders with data quality issues
Row(Order_ID="O2", Order_Date="21/8/2016", Customer_ID=None, Product_ID="P1", Profit=100.0)  # Null customer
Row(Order_ID="O3", Order_Date="21/8/2016", Customer_ID="C1", Product_ID="P1", Profit=-50.0)  # Negative profit

# Products with invalid prices
Row(Product_ID="P2", Category="Tech", Sub_Category="Phones", Price_per_product=0.0)  # Zero price
Row(Product_ID="P3", Category="Tech", Sub_Category="Phones", Price_per_product=-100.0)  # Negative price
```

## Coverage Metrics

The extended test suite provides comprehensive coverage across:

- **Function Coverage**: All public functions tested
- **Branch Coverage**: All conditional logic paths tested
- **Edge Case Coverage**: Boundary conditions and error scenarios
- **Integration Coverage**: End-to-end workflow validation
- **Performance Coverage**: Large dataset and scalability testing

## Best Practices

### Writing New Tests
1. Use descriptive test names that explain the scenario
2. Follow the Arrange-Act-Assert pattern
3. Test both positive and negative cases
4. Include edge cases and boundary conditions
5. Use appropriate test fixtures for data setup
6. Verify both expected results and error conditions

### Test Data Management
1. Use small, focused datasets for unit tests
2. Create realistic data for integration tests
3. Include edge cases in test data
4. Clean up test tables after execution
5. Use consistent naming conventions

### Error Testing
1. Test all error conditions explicitly
2. Verify error messages are meaningful
3. Ensure proper exception types are raised
4. Test error handling at different pipeline stages

## Troubleshooting

### Common Issues

1. **Spark Session Conflicts**
   ```python
   # Ensure proper session management in tests
   @pytest.fixture(scope="session")
   def spark():
       return SparkSession.builder.master("local[2]").getOrCreate()
   ```

2. **Table Already Exists Errors**
   ```python
   # Use overwrite mode in tests
   df.write.format("delta").mode("overwrite").saveAsTable("test.table_name")
   ```

3. **Path Issues**
   ```python
   # Add project root to Python path
   sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
   ```

### Performance Considerations
- Use `local[2]` for Spark master in tests (limits resource usage)
- Keep test datasets reasonably sized
- Use appropriate test markers for slow tests
- Consider parallel test execution for large test suites

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Add appropriate test markers
3. Update this README if adding new test categories
4. Ensure tests are deterministic and repeatable
5. Include both positive and negative test cases

## Test Markers

The test suite uses pytest markers for categorization:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.extended` - Extended coverage tests
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.performance` - Performance and scalability tests
