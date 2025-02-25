import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table } from "@/components/ui/table";
import axios from "axios";

export default function LandingPage() {
  const [waiters, setWaiters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [futureTips, setFutureTips] = useState([]);

  useEffect(() => {
    fetchWaiterData();
  }, []);

  const fetchWaiterData = async () => {
    try {
      const response = await axios.get("http://127.0.0.1:5000/api/waiters", {
        headers: {
          "Content-Type": "application/json"
        }
      });
      if (response.data) {
        setWaiters(response.data.waiters || []);
        setFutureTips(response.data.future_tips || []);
      } else {
        console.error("Unexpected API response structure:", response.data);
      }
    } catch (error) {
      console.error("Error fetching waiter data:", error.response ? error.response.data : error.message);
      alert("Failed to fetch waiter data. Please check if the server is running and CORS is enabled.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-purple-300 to-pink-300 flex flex-col items-center justify-center p-10">
      <h1 className="text-5xl font-bold text-white drop-shadow-lg">Waiter2Go!</h1>
      <p className="text-lg text-white mt-4 opacity-80">
        A workflow management app that rewards waiters for good service!
      </p>
      <Button className="mt-6 bg-white text-purple-500" onClick={fetchWaiterData}>
        Refresh Data
      </Button>

      {/* Waiter Ranking System */}
      <div className="mt-10 w-full max-w-4xl">
        <h2 className="text-3xl font-bold text-white mb-4 text-center">Waiter Rankings</h2>
        {loading ? (
          <p className="text-white text-center">Loading...</p>
        ) : (
          <Table className="bg-white shadow-lg rounded-lg overflow-hidden">
            <thead className="bg-purple-500 text-white">
              <tr>
                <th className="p-3">Waiter UUID</th>
                <th className="p-3">Total Tips</th>
                <th className="p-3">Revenue</th>
                <th className="p-3">Performance Cluster</th>
              </tr>
            </thead>
            <tbody>
              {waiters.map((waiter, index) => (
                <tr key={index} className="text-center border-b">
                  <td className="p-3">{waiter.waiter_uuid || "N/A"}</td>
                  <td className="p-3">${waiter.total_tips?.toFixed(2) || "0.00"}</td>
                  <td className="p-3">${waiter.total_revenue?.toFixed(2) || "0.00"}</td>
                  <td className="p-3 font-bold text-blue-600">{waiter.performance_cluster ?? "N/A"}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </div>

      {/* Future Tip Predictions */}
      <div className="mt-10 w-full max-w-4xl">
        <h2 className="text-3xl font-bold text-white mb-4 text-center">Future Tip Predictions</h2>
        {loading ? (
          <p className="text-white text-center">Loading...</p>
        ) : (
          <Table className="bg-white shadow-lg rounded-lg overflow-hidden">
            <thead className="bg-purple-500 text-white">
              <tr>
                <th className="p-3">Future Date</th>
                <th className="p-3">Predicted Tip %</th>
              </tr>
            </thead>
            <tbody>
              {futureTips.map((tip, index) => (
                <tr key={index} className="text-center border-b">
                  <td className="p-3">{`Prediction ${index + 1}`}</td>
                  <td className="p-3 font-bold text-green-600">{tip[0]?.toFixed(2) || "0.00"}%</td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </div>
    </div>
  );
}
