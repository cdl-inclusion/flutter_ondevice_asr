package com.example.example

import androidx.test.rule.ActivityTestRule
import dev.flutter.plugins.integration_test.FlutterTestRunner
import org.junit.Rule
import org.junit.runner.RunWith

@RunWith(FlutterTestRunner::class)
public class MainActivityTest {
    @Rule
    @JvmField
    var rule: ActivityTestRule<MainActivity?> =
        ActivityTestRule(MainActivity::class.java, true, false)
}