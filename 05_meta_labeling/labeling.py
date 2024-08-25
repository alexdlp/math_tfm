# Funciones del libro 'Advances in Financial Machine Learning' adaptadas para que tengan puto sentido.


import pandas as pd
import numpy as np
import multiprocessing as mp
import time
from typing import Callable, List, Dict, Any
import datetime as dt
import sys
import time

# HUMBRALES DINÁMICOS
# # función del libro de lopez de prado para obtener la volatilidad intradía
# def getDailyVol(close,span0=100):
#     # daily vol, reindexed to close
#     df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
#     df0=df0[df0>0]
#     df0=pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
#     df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
#     df0=df0.ewm(span=span0).std()
#     return df0

# función reescrita para humanos.
# se han añadido comprovaciones (que exista la columna close, y que el indice sea del tipo datetime)
def calculate_daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """
    Calculate the daily volatility as an exponentially weighted standard deviation of returns.
    
    Args:
        close (pd.Series): Series of closing prices with datetime index.
        span (int): Span parameter for the exponentially weighted moving standard deviation.
    
    Returns:
        pd.Series: Series of daily volatilities reindexed to match the closing prices.
    """
    # Check if the index is a datetime index, if not, try to convert it
    if not pd.api.types.is_datetime64_any_dtype(close.index):
        try:
            close.index = pd.to_datetime(close.index)
        except Exception as e:
            raise TypeError("Index cannot be converted to datetime. Please ensure it is a datetime index.") from e

    # Find indices of the previous day's close prices
    previous_day_indices = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    previous_day_indices = previous_day_indices[previous_day_indices > 0]
    previous_day_dates = pd.Series(close.index[previous_day_indices - 1], index=close.index[close.shape[0] - previous_day_indices.shape[0]:])
    
    # Calculate daily returns
    daily_returns = close.loc[previous_day_dates.index] / close.loc[previous_day_dates.values].values - 1
    
    # Calculate exponentially weighted moving standard deviation of returns
    daily_volatility = daily_returns.ewm(span=span).std()
    
    return daily_volatility


# def applyPtSlOnT1(close,events,ptSl,molecule):
#     # apply stop loss/profit taking, if it takes place before t1 (end of event)
#     events_=events.loc[molecule]
#     out=events_[['t1']].copy(deep=True)
#     if ptSl[0]>0:
#         pt=ptSl[0]*events_['trgt']
#     else:
#         pt=pd.Series(index=events.index) # NaNs
#     if ptSl[1]>0:
#         sl=-ptSl[1]*events_['trgt']
#     else:
#         sl=pd.Series(index=events.index) # NaNs
#         for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
#             df0=close[loc:t1] # path prices
#             df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
#             out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss.
#             out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
#     return out

def apply_profit_taking_stop_loss_on_t1(close: pd.Series, events: pd.DataFrame, ptSl: list, molecule: list) -> pd.DataFrame:
    """
    Apply profit-taking and stop-loss barriers to financial events and determine when each barrier is touched.

    Args:
        close (pd.Series): Series of closing prices with a datetime index.
        events (pd.DataFrame): DataFrame containing event data with columns:
            - 't1': Timestamp for the vertical barrier. NaN means no vertical barrier.
            - 'trgt': Target width for horizontal barriers.
        ptSl (list): List with two non-negative float values:
            - ptSl[0]: Factor for setting the width of the upper profit-taking barrier.
            - ptSl[1]: Factor for setting the width of the lower stop-loss barrier.
        molecule (list): List of indices from the events DataFrame to process.

    Returns:
        pd.DataFrame: DataFrame with timestamps for when each barrier was touched, including:
            - 'stop_loss_time': Timestamp when the stop-loss barrier was first touched.
            - 'profit_taking_time': Timestamp when the profit-taking barrier was first touched.
    """
    
    # Select the subset of events based on the provided indices
    events_ = events.loc[molecule]
    
    # Initialize the output DataFrame with the 't1' column
    result_df = events_[['t1']].copy(deep=True)
    
    # Calculate the upper profit-taking barrier if the factor is greater than 0
    if ptSl[0] > 0:
        upper_barrier = ptSl[0] * events_['trgt']
    else:
        upper_barrier = pd.Series(index=events_.index)  # NaNs if no upper barrier
    
    # Calculate the lower stop-loss barrier if the factor is greater than 0
    if ptSl[1] > 0:
        lower_barrier = -ptSl[1] * events_['trgt']
    else:
        lower_barrier = pd.Series(index=events_.index)  # NaNs if no lower barrier
    
    # Process each event to determine when barriers are touched
    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        # Extract the price path from the event time to the vertical barrier time
        price_path = close[loc:t1]
        
        # Calculate the returns based on the price path
        returns = (price_path / close[loc] - 1) * events_.at[loc, 'side']
        
        # Determine the earliest touch time for the stop-loss barrier
        if not lower_barrier.empty:
            stop_loss_touch_time = returns[returns < lower_barrier[loc]].index.min()
        else:
            stop_loss_touch_time = pd.NaT
        
        # Determine the earliest touch time for the profit-taking barrier
        if not upper_barrier.empty:
            profit_taking_touch_time = returns[returns > upper_barrier[loc]].index.min()
        else:
            profit_taking_touch_time = pd.NaT
        
        # Store the results
        result_df.loc[loc, 'stop_loss_time'] = stop_loss_touch_time
        result_df.loc[loc, 'profit_taking_time'] = profit_taking_touch_time
    
    return result_df


def get_events(close, t_events, pt_sl, trgt, min_ret, num_threads, t1=False):
    """
    Finds the time of the first barrier touch for each event.
    
    Args:
    close (pd.Series): Series of closing prices with datetime index.
    t_events (pd.DatetimeIndex): Timestamps for events to be evaluated.
    pt_sl (float): Width of the horizontal barriers. If 0, the barrier is disabled.
    trgt (pd.Series): Target returns for each event.
    min_ret (float): Minimum target return required to perform barrier search.
    num_threads (int): Number of threads for parallel processing.
    t1 (pd.Series or bool): Timestamps for vertical barriers. If False, vertical barriers are disabled.
    
    Returns:
    pd.DataFrame: DataFrame containing timestamps of the first barrier touches.
    """
    # 1. Filter targets (trgt) to include only those above the minimum return (min_ret)
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    
    # 2. Handle vertical barriers (t1)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)
    
    # 3. Create the events DataFrame with initial values
    side_ = pd.Series(1., index=trgt.index)  # Default side value
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    
    # 4. Apply profit-taking and stop-loss barriers
    df0 = parallelize_pandas_function(func=apply_profit_taking_stop_loss_on_t1, pdObj=('molecule', events.index),
                      numThreads=num_threads, close=close, events=events, ptSl=[pt_sl, pt_sl])
    
    # 5. Update events DataFrame with the first barrier touch times
    events['t1'] = df0.dropna(how='all').min(axis=1)  # First touch time
    events = events.drop('side', axis=1)  # Drop the 'side' column
    
    return events


# def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs): 
#     ''' 
#     Parallelize jobs, return a DataFrame or Series 
#     + func: function to be parallelized. Returns a DataFrame 
#     + pdObj[0]: Name of argument used to pass the molecule 
#     + pdObj[1]: List of atoms that will be grouped into molecules 
#     + kargs: any other argument needed by func
#     '''
#     import pandas as pd 
#     if linMols:
#         parts=linParts(len(pdObj[1]),numThreads*mpBatches) 
#     else:
#         parts=nestedParts(len(pdObj[1]),numThreads*mpBatches) 
#     jobs=[] 
#     for i in xrange(1,len(parts)): 
#         job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func} 
#         job.update(kargs) 
#         jobs.append(job) 
#     if numThreads==1:
#         out=processJobs_(jobs) 
#     else:
#         out=processJobs(jobs,numThreads=numThreads) 
#     if isinstance(out[0],pd.DataFrame):
#         df0=pd.DataFrame() 
#     elif isinstance(out[0],pd.Series):
#         df0=pd.Series() 
#     else:
#         return out 
#     for i in out:
#         df0=df0.append(i) 
#     return df0.sort_index()

def parallelize_pandas_function(func, partition_info, num_threads=24, num_batches=1, use_linear_partitions=True, **kwargs):
    """
    Parallelize the execution of a function that processes pandas DataFrames or Series.

    This function is an updated version of the original mpPandasObj, designed to handle
    parallel processing of pandas objects. It distributes the data into smaller partitions
    and executes the provided function across multiple threads.

    Parameters:
    func (callable): The function to be executed in parallel. It should process pandas DataFrames
                     or Series and return a DataFrame or Series.
    partition_info (tuple): A tuple where:
        - partition_info[0] (str): The name of the argument used to pass the data partition to the function.
        - partition_info[1] (list): A list of data partitions to be grouped into molecules for parallel processing.
    num_threads (int, optional): The number of threads to use for parallel processing. Default is 24.
    num_batches (int, optional): The number of parallel batches (jobs per core). Default is 1.
    use_linear_partitions (bool, optional): Whether to use linear partitioning. Default is True.
    **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
    pd.DataFrame or pd.Series: The combined results from all parallel executions, sorted by index.
    """
    if use_linear_partitions:
        partition_indices = linear_partitions(len(partition_info[1]), num_threads * num_batches)
    else:
        partition_indices = nested_partitions(len(partition_info[1]), num_threads * num_batches)
        
    jobs = []
    for i in range(1, len(partition_indices)):
        job = {
            partition_info[0]: partition_info[1][partition_indices[i-1]:partition_indices[i]],
            'func': func
        }
        job.update(kwargs)
        jobs.append(job)
        
    if num_threads == 1:
        results = processJobs_(jobs)
    else:
        results = processJobs(jobs, numThreads=num_threads)
    
    if isinstance(results[0], pd.DataFrame):
        combined_results = pd.DataFrame()
    elif isinstance(results[0], pd.Series):
        combined_results = pd.Series()
    else:
        return results
    
    for result in results:
        combined_results = combined_results.append(result)
    
    return combined_results.sort_index()


def linear_partitions(num_atoms: int, num_threads: int) -> np.ndarray:
    """
    Partition a list of atoms into approximately equal-sized groups using a linear partitioning approach.

    This function is a descendant of the 'linParts' function and is designed to divide a total number
    of atoms into a specified number of partitions or groups. The number of partitions should not exceed
    the total number of atoms.

    Args:
        num_atoms (int): Total number of atoms to be divided into partitions.
        num_threads (int): Number of partitions (or groups) to divide the atoms into.

    Returns:
        np.ndarray: Array of partition indices defining the start and end points of each group.
    """
    # Compute the boundaries of each partition
    partition_boundaries = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    
    # Round up the partition boundaries to ensure all atoms are included
    partition_boundaries = np.ceil(partition_boundaries).astype(int)
    
    return partition_boundaries

def nested_partitions(total_atoms: int, number_of_threads: int, upper_triangle: bool = False) -> np.ndarray:
    """
    Partition a list of atoms into groups using a nested partitioning approach.

    This function is a descendant of the 'nestedParts' function and is designed to create partitions
    where the number of atoms in each group may vary, with an optional upper triangular weighting for the partitions.

    Args:
        total_atoms (int): Total number of atoms to be divided into partitions.
        number_of_threads (int): Number of partitions (or groups) to divide the atoms into.
        upper_triangle (bool): If True, applies upper triangular weighting to the partitions.

    Returns:
        np.ndarray: Array of partition indices defining the start and end points of each group.
    """
    # Initialize the list of partition boundaries with the starting point
    partition_boundaries = [0]
    
    # Compute the number of partitions
    effective_threads = min(number_of_threads, total_atoms)
    
    for _ in range(effective_threads):
        # Calculate the partition boundary based on the quadratic formula
        next_partition = 1 + 4 * (partition_boundaries[-1]**2 + partition_boundaries[-1] +
                                   total_atoms * (total_atoms + 1) / number_of_threads)
        next_partition = (-1 + np.sqrt(next_partition)) / 2
        partition_boundaries.append(next_partition)
    
    # Round the partition boundaries to the nearest integer
    partition_boundaries = np.round(partition_boundaries).astype(int)
    
    if upper_triangle:
        # Apply upper triangular weighting to the partitions
        partition_boundaries = np.cumsum(np.diff(partition_boundaries)[::-1])
        partition_boundaries = np.append(np.array([0]), partition_boundaries)
    
    return partition_boundaries

def processJobs_(jobs: List[Dict[str, Any]]) -> List[Any]:
    """
    Execute a list of jobs sequentially, useful for debugging purposes.

    Each job in the list must contain a 'func' key, which is a callback function to be executed.

    Args:
        jobs (List[Dict[str, Any]]): List of dictionaries where each dictionary contains a 'func' key with the function to execute.

    Returns:
        List[Any]: A list of results from processing each job.
    """
    results = []
    for job in jobs:
        result = expand_callback(job)
        results.append(result)
    return results

def processJobs(jobs: List[Dict[str, Any]], task: str = None, numThreads: int = 24) -> List[Any]:
    """
    Execute a list of jobs in parallel using multiprocessing.

    Each job in the list must contain a 'func' key, which is a callback function to be executed.

    Args:
        jobs (List[Dict[str, Any]]): List of dictionaries where each dictionary contains a 'func' key with the function to execute.
        task (str, optional): Name of the task to be used for progress reporting. If None, uses the function name from the first job.
        numThreads (int): Number of threads (processes) to use for parallel execution.

    Returns:
        List[Any]: A list of results from processing each job.
    """
    if task is None:
        task = jobs[0]['func'].__name__
    
    # Create a pool of worker processes
    with mp.Pool(processes=numThreads) as pool:
        # Start processing jobs in parallel
        outputs = pool.imap_unordered(expand_callback, jobs)
        results = []
        start_time = time.time()

        # Process results as they become available and report progress
        for i, result in enumerate(outputs, start=1):
            results.append(result)
            report_progress(i, len(jobs), start_time, task)
    
    return results

def expand_callback(kargs: dict) -> Any:
    """
    Expand the arguments of a callback function and execute it.

    The function is retrieved from the 'func' key in the kargs dictionary, 
    and the remaining keys in the dictionary are passed as keyword arguments.

    Args:
        kargs (dict): Dictionary containing:
            - 'func': The callback function to execute.
            - Any additional keyword arguments for the function.

    Returns:
        Any: The result of the function execution.
    """
    callback_function = kargs['func']
    # Remove 'func' key to avoid passing it as an argument to the callback function
    del kargs['func']
    # Execute the callback function with the remaining arguments
    result = callback_function(**kargs)
    return result

def report_progress(current_job_num: int, total_jobs: int, start_time: float, task_name: str) -> None:
    """
    Report the progress of asynchronous jobs as they are completed.

    Args:
        current_job_num (int): The number of the current job being processed.
        total_jobs (int): The total number of jobs.
        start_time (float): The timestamp when the job processing started.
        task_name (str): The name of the task being performed.

    Returns:
        None
    """
    # Calculate progress metrics
    progress_ratio = float(current_job_num) / total_jobs
    elapsed_time_minutes = (time.time() - start_time) / 60
    remaining_time_minutes = elapsed_time_minutes * (1 / progress_ratio - 1) if progress_ratio > 0 else 0

    # Create timestamp for reporting
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))
    
    # Format and print the progress message
    progress_message = (
        f"{time_stamp} {round(progress_ratio * 100, 2)}% {task_name} done after "
        f"{round(elapsed_time_minutes, 2)} minutes. Remaining "
        f"{round(remaining_time_minutes, 2)} minutes."
    )
    
    # Output the progress message
    if current_job_num < total_jobs:
        sys.stderr.write(progress_message + '\r')
    else:
        sys.stderr.write(progress_message + '\n')

def label_observations(events: pd.DataFrame, close_prices: pd.Series) -> pd.DataFrame:
    """
    Label observations based on the events and closing prices.

    This function is a continuation of the functionality provided by the `getBins` function. 
    It labels each observation by calculating the realized return at the time of the first barrier touched 
    and assigns a label based on the sign of the return.

    Args:
        events (pd.DataFrame): DataFrame containing event timestamps with columns:
            - 't1': The timestamp of the vertical barrier touched.
            - Any other relevant columns.
        close_prices (pd.Series): Series containing the closing prices of the instrument.
    
    Returns:
        pd.DataFrame: DataFrame with two columns:
            - 'ret': The return realized at the time of the first touched barrier.
            - 'bin': The label {-1, 0, 1} indicating the sign of the return.
    """
    # 1) Align prices with event timestamps
    events_with_vertical_barrier = events.dropna(subset=['t1'])
    all_timestamps = events_with_vertical_barrier.index.union(events_with_vertical_barrier['t1'].values).drop_duplicates()
    aligned_prices = close_prices.reindex(all_timestamps, method='bfill')
    
    # 2) Create the output DataFrame
    labeled_data = pd.DataFrame(index=events_with_vertical_barrier.index)
    labeled_data['ret'] = aligned_prices.loc[events_with_vertical_barrier['t1'].values].values / aligned_prices.loc[events_with_vertical_barrier.index] - 1
    labeled_data['bin'] = np.sign(labeled_data['ret'])
    
    return labeled_data
