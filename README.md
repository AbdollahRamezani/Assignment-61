#Install the required dependencies :
 ```
pip install -r requirements.txt
``` 
### without Dropout layer
<table>
    <tr>
        <td>Dimension</td>
        <td>Train Loss</td>
        <td>Train Accuracy</td>
        <td>Test Loss</td>
        <td>Test Accuracy</td>
        <td>Inference Time</td>
    </tr>
    <tr>
        <td>50d</td>
        <td>0.495</td>
        <td>0.901</td>
        <td>0.569</td>
        <td>0.839</td>
        <td>0.00058</td>
    </tr>    
    </tr>
        <td>100d</td>
        <td>0.521</td>
        <td>0.871</td>
        <td>0.668</td>
        <td>0.803</td>
        <td>0.00173</td>
    </tr>
    <tr>
        <td>200d</td>
        <td>0.314</td>
        <td>0.954</td>
        <td>0.526</td>
        <td>0.821</td>
        <td>0.0012</td>
    </tr>
    <tr>
        <td>300d</td>
        <td>0.224</td>
        <td>0.984</td>
        <td>0.461</td>
        <td>0.875</td>
        <td>0.00098</td>
    </tr>
   
</table>

<br>

### with Dropout layer
<table>
    <tr>
        <td>Dimension</td>
        <td>Train Loss</td>
        <td>Train Accuracy</td>
        <td>Test Loss</td>
        <td>Test Accuracy</td>
        <td>Inference Time</td>
    </tr>
    <tr>
        <td>50d</td>
        <td>0.960</td>
        <td>0.651</td>
        <td>0.957</td>
        <td>0.732</td>
        <td>0.0012</td>
    </tr>    
    </tr>
        <td>100d</td>
        <td>0.521</td>
        <td>0.871</td>
        <td>0.668</td>
        <td>0.803</td>
        <td>0.00173</td>
    </tr>
    <tr>
        <td>200d</td>
        <td>0.533</td>
        <td>0.871</td>
        <td>0.632</td>
        <td>0.857</td>
        <td>0.0010</td>
    </tr>
    <tr>
        <td>300d</td>
        <td>0.409</td>
        <td>0.916</td>
        <td>0.542</td>
        <td>0.875</td>
        <td>0.0011</td>
    </tr>
   
</table>