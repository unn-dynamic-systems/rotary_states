cd ..
    echo "spawn custom_integrator"
    time poetry run python examples/custom_integrator.py &
    echo "spawn odeint"
    time poetry run python examples/odeint.py &
    echo "wait ..."
wait
