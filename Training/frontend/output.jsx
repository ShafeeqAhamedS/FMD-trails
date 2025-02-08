
import { useState, useEffect } from "react"
import axios from "axios"

const API_BASE_URL = "http://localhost:8000"

const App = () => {
  const [inputs, setInputs] = useState({ Position: "", Level: "" })  // Adjusted based on the model inputs from JSON Code block
  const [prediction, setPrediction] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [healthStatus, setHealthStatus] = useState("Checking...")
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    checkHealth()
    fetchMetrics()
  }, [])

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`)
      setHealthStatus(response.data.status === "ok" ? "Online" : "Offline")
    } catch (err) {
      setHealthStatus("Offline")
      setError("Failed to connect to the backend. Please ensure the server is running.")
    }
  }

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/metrics`)
      setMetrics(response.data)
    } catch (err) {
      setMetrics(null)
      console.error("Failed to fetch metrics:", err);
    }
  }

  const handleInputChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setPrediction(null)
    setLoading(true)

    // Convert inputs to the appropriate data types (float, int, etc.)
    const convertedInputs = {
      Position: parseInt(inputs.Position),
      Level: parseInt(inputs.Level),  // only If applicable
    }

    // Validate inputs
    if (isNaN(convertedInputs.Position) || isNaN(convertedInputs.Level)) {
      setError("Please enter valid numbers for Position and Level.");
      setLoading(false);
      return;
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, { inputs: convertedInputs })
      setPrediction(response.data.predictions)
    } catch (err) {
      setError("Prediction failed. Please check your input and try again.");
      console.error("Prediction failed:", err);
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-indigo-100 py-6 sm:py-8 md:py-10 lg:py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-xs sm:max-w-md md:max-w-lg lg:max-w-xl xl:max-w-2xl mx-auto bg-white rounded-xl shadow-md overflow-hidden">
        <div className="p-6 sm:p-8 md:p-10">
          <div className="uppercase tracking-wide text-sm text-indigo-500 font-semibold mb-1">Salary Prediction</div>
          <h1 className="text-lg sm:text-xl md:text-2xl lg:text-3xl font-medium text-black mb-2">
            Salary Prediction Model {/* Update the title */}
          </h1>
          <p className="mt-2 text-sm sm:text-base text-gray-500">
            Predict salary based on position and level. {/* Update description */}
          </p>

          <div className="mt-4 flex items-center">
            <span className="text-sm sm:text-base text-gray-700 mr-2">API Status:</span>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs sm:text-sm font-medium ${
                healthStatus === "Online" ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
              }`}
            >
              {healthStatus}
            </span>
          </div>

          {error && (
            <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
              <strong className="font-bold">Error:</strong>
              <span className="block sm:inline">{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="mt-6">
            <label className="block">
              <span className="text-sm sm:text-base text-gray-700">Position: </span>
              <input
                type="number"
                name="Position"
                value={inputs.Position}
                onChange={handleInputChange}
                className="rounded-md mt-1 block w-full px-3 py-2 border border-gray-300 shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo"
                required
              />
            </label>

            <label className="block mt-4">
              <span className="text-sm sm:text-base text-gray-700">Level: </span>
              <input
                type="number"
                name="Level"
                value={inputs.Level}
                onChange={handleInputChange}
                className="rounded-md mt-1 block w-full px-3 py-2 border border-gray-300 shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo"
                required
              />
            </label>

            <button
              type="submit"
              className="mt-4 w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50 transition duration-150 ease-in-out text-sm sm:text-base"
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Predicting...
                </span>
              ) : (
                "Predict"
              )}
            </button>
          </form>

          {prediction && (
            <div className="mt-6">
              <p className="text-xl sm:text-2xl font-semibold text-gray-800">
                Predicted Salary: {prediction[0]} {/* Update based on response & response model */}
              </p>
            </div>
          )}

          {metrics && (
            <div className="mt-8">
              <h2 className="text-base sm:text-lg md:text-xl font-semibold text-gray-800 mb-4">
                Model Metrics
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {metrics.mse && (
                  <div className="bg-indigo-100 p-4 rounded-md">
                    <h3 className="text-xs sm:text-sm font-medium text-indigo-800">MSE</h3>
                    <p className="mt-1 text-lg sm:text-xl md:text-2xl font-semibold text-indigo-900">
                      {metrics.mse.toFixed(4)}
                    </p>
                  </div>
                )}
                {metrics.r2 && (
                  <div className="bg-indigo-100 p-4 rounded-md">
                    <h3 className="text-xs sm:text-sm font-medium text-indigo-800">R2 Score</h3>
                    <p className="mt-1 text-lg sm:text-xl md:text-2xl font-semibold text-indigo-900">
                      {metrics.r2.toFixed(4)}
                    </p>
                  </div>
                )}
              </div>
              {!metrics.mse && !metrics.r2 && (
                <p className="text-gray-500">No metrics available.</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
