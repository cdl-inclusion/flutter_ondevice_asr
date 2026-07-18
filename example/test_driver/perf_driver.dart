import 'package:integration_test/integration_test_driver.dart';

Future<void> main() => integrationDriver(
  responseDataCallback: (data) =>
      writeResponseData({'traceEvents': data?['timeline']['traceEvents']},
      testOutputFilename: 'performance_trace'),
);
