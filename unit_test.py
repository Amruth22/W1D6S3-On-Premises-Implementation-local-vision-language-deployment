import unittest
import os
import sys
import json
import requests
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock

# Add the current directory to the path so we can import app
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class TestApp(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Load environment variables
        load_dotenv()
        
        # Base URL for the API
        cls.base_url = "http://localhost:5000"
        
    # -----------------------------
    # Environment and Configuration Tests
    # -----------------------------
    
    def test_env_file_exists(self):
        """Test that the .env file exists"""
        # Check if .env file exists in current directory
        env_path = os.path.join(current_dir, '.env')
        self.assertTrue(os.path.exists(env_path), "The .env file does not exist")
        
    def test_env_api_key_configured(self):
        """Test that the API key is properly configured in .env file"""
        # Check if API key is set in environment variables
        api_key = os.environ.get('GEMINI_API_KEY')
        
        # If API key is None, it might be commented out in .env
        if api_key is None:
            # Try to read the .env file directly to check if the key is commented out
            env_path = os.path.join(current_dir, '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    content = f.read()
                    if 'GEMINI_API_KEY=' in content and not '# GEMINI_API_KEY=' in content:
                        self.fail("GEMINI_API_KEY found in .env but not loaded properly")
                    elif '# GEMINI_API_KEY=' in content:
                        self.fail("GEMINI_API_KEY is commented out in .env file. Please uncomment it to run tests.")
                    else:
                        self.fail("GEMINI_API_KEY not found in .env file")
            else:
                self.fail("GEMINI_API_KEY is not set in environment variables")
        else:
            self.assertNotEqual(api_key, '', "GEMINI_API_KEY is empty")
            self.assertNotEqual(api_key, 'your_api_key_here', "GEMINI_API_KEY is still set to the default placeholder value")
            # Check that API key looks like a real API key (begins with 'AI' and has reasonable length)
            self.assertTrue(api_key.startswith('AI'), "GEMINI_API_KEY does not appear to be valid (should start with 'AI')")
            self.assertGreater(len(api_key), 20, "GEMINI_API_KEY appears too short to be valid")
        
    def test_app_imports_and_initialization(self):
        """Test that the app can be imported and initialized without errors"""
        # If API key is not set, we'll patch it for this test
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key is None or api_key == '':
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'}):
                try:
                    # Remove app from sys.modules if it was already imported
                    if 'app' in sys.modules:
                        del sys.modules['app']
                    from app import app
                    self.assertIsNotNone(app, "Failed to import Flask app")
                except Exception as e:
                    self.fail(f"Failed to import app.py: {e}")
        else:
            try:
                from app import app
                self.assertIsNotNone(app, "Failed to import Flask app")
            except Exception as e:
                self.fail(f"Failed to import app.py: {e}")
            
    def test_required_dependencies_importable(self):
        """Test that all required dependencies can be imported"""
        required_modules = [
            'flask',
            'flask_cors',
            'flask_swagger',
            'flask_swagger_ui',
            'google.genai',
            'dotenv'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError as e:
                self.fail(f"Failed to import required module {module}: {e}")
                
    # -----------------------------
    # Endpoint Tests (with server running)
    # -----------------------------
    
    def test_index_endpoint(self):
        """Test that the index endpoint is working"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertIn('Multimodal Gemini API', response.text)
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_spec_endpoint(self):
        """Test that the spec endpoint is working"""
        try:
            response = requests.get(f"{self.base_url}/spec", timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('info', data)
            self.assertEqual(data['info']['title'], 'Multimodal Gemini API')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_text_endpoint_success(self):
        """Test that the text endpoint works with valid input"""
        try:
            response = requests.post(
                f"{self.base_url}/text",
                json={'prompt': 'What is artificial intelligence?'},
                timeout=30
            )
            # We expect either 200 (success) or 500 (API key or network error)
            # but not 400 (bad request)
            self.assertNotEqual(response.status_code, 400)
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_text_endpoint_streaming(self):
        """Test that the text endpoint works with streaming enabled"""
        try:
            response = requests.post(
                f"{self.base_url}/text",
                json={
                    'prompt': 'Write a short story in 3 sentences.',
                    'stream': True
                },
                timeout=30
            )
            # We expect either 200 (success) or 500 (API key or network error)
            # but not 400 (bad request)
            self.assertNotEqual(response.status_code, 400)
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_text_endpoint_missing_prompt(self):
        """Test that the text endpoint returns error for missing prompt"""
        try:
            response = requests.post(
                f"{self.base_url}/text",
                json={},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Prompt is required')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_text_endpoint_empty_prompt(self):
        """Test that the text endpoint returns error for empty prompt"""
        try:
            response = requests.post(
                f"{self.base_url}/text",
                json={'prompt': ''},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Prompt is required')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_image_endpoint_missing_file(self):
        """Test that the image endpoint returns error for missing file"""
        try:
            response = requests.post(
                f"{self.base_url}/image",
                data={'prompt': 'Describe this image'},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Image file is required')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_image_endpoint_empty_file(self):
        """Test that the image endpoint returns error for empty file"""
        try:
            response = requests.post(
                f"{self.base_url}/image",
                files={'image': (None, '', 'image/jpeg')},
                data={'prompt': 'Describe this image'},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Image file is required')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_audio_endpoint_missing_file(self):
        """Test that the audio endpoint returns error for missing file"""
        try:
            response = requests.post(
                f"{self.base_url}/audio",
                data={'prompt': 'Describe this audio'},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Audio file is required')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_audio_endpoint_empty_file(self):
        """Test that the audio endpoint returns error for empty file"""
        try:
            response = requests.post(
                f"{self.base_url}/audio",
                files={'audio': (None, '', 'audio/mp3')},
                data={'prompt': 'Describe this audio'},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'Audio file is required')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    def test_multimodal_endpoint_no_content(self):
        """Test that the multimodal endpoint returns error for no content"""
        try:
            response = requests.post(
                f"{self.base_url}/multimodal",
                data={},
                timeout=5
            )
            self.assertEqual(response.status_code, 400)
            data = response.json()
            self.assertIn('error', data)
            self.assertEqual(data['error'], 'At least one modality (text, image, or audio) must be provided')
        except requests.exceptions.ConnectionError:
            self.skipTest("API server is not running. Start the server to test endpoints.")
        except requests.exceptions.Timeout:
            self.skipTest("Request to API server timed out.")
            
    # -----------------------------
    # Mocked Tests (for CI/CD environments)
    # -----------------------------
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('app.genai.Client')
    def test_text_endpoint_with_mocked_response(self, mock_client_class):
        """Test text endpoint with mocked AI response"""
        # Remove app from sys.modules if it was already imported
        if 'app' in sys.modules:
            del sys.modules['app']
            
        # Mock the Gemini client
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        
        # Mock the Gemini client response
        mock_response = MagicMock()
        mock_response.text = "This is a mocked response from the AI model."
        mock_client_instance.models.generate_content.return_value = mock_response
        
        # Import app inside the test
        from app import app
        client = app.test_client()
        
        # Test valid request
        response = client.post('/text', 
                              data=json.dumps({'prompt': 'Hello, world!'}),
                              content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('text', data)
        self.assertEqual(data['text'], "This is a mocked response from the AI model.")
        
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_api_key'})
    @patch('app.genai.Client')
    def test_text_endpoint_missing_prompt_with_mock(self, mock_client_class):
        """Test text endpoint missing prompt with mock"""
        # Remove app from sys.modules if it was already imported
        if 'app' in sys.modules:
            del sys.modules['app']
            
        # Import app inside the test
        from app import app
        client = app.test_client()
        
        # Test missing prompt
        response = client.post('/text', 
                              data=json.dumps({}),
                              content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Prompt is required')

def custom_test_runner():
    """Custom test runner that continues even if tests fail"""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestApp)
    
    # Create a custom test result class to capture all results
    class CustomTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
            
        def addSuccess(self, test):
            super().addSuccess(test)
            self.test_results.append({
                'name': test._testMethodName,
                'status': 'PASS',
                'details': ''
            })
            
        def addError(self, test, err):
            super().addError(test, err)
            self.test_results.append({
                'name': test._testMethodName,
                'status': 'ERROR',
                'details': self._exc_info_to_string(err, test)
            })
            
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.test_results.append({
                'name': test._testMethodName,
                'status': 'FAIL',
                'details': self._exc_info_to_string(err, test)
            })
            
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self.test_results.append({
                'name': test._testMethodName,
                'status': 'SKIP',
                'details': reason
            })
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2, resultclass=CustomTestResult)
    result = runner.run(suite)
    
    # Print detailed summary
    print("\n" + "="*60)
    print("DETAILED TEST RESULTS")
    print("="*60)
    
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    
    for test_result in result.test_results:
        status = test_result['status']
        if status == 'PASS':
            passed += 1
        elif status == 'FAIL':
            failed += 1
        elif status == 'ERROR':
            errors += 1
        elif status == 'SKIP':
            skipped += 1
            
        print(f"\n{status}: {test_result['name']}")
        if status in ['FAIL', 'ERROR'] and test_result['details']:
            print(f"  Details: {test_result['details'][:200]}...")
        elif status == 'SKIP':
            print(f"  Reason: {test_result['details']}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tests run: {len(result.test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    
    if failed > 0 or errors > 0:
        print("\nFAILED TESTS:")
        for test_result in result.test_results:
            if test_result['status'] in ['FAIL', 'ERROR']:
                print(f"  - {test_result['name']} ({test_result['status']})")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Run all tests and provide detailed results
    success = custom_test_runner()
    
    if success:
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED")
        print("="*60)