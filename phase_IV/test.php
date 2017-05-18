<html>
  <head>
  <title>PHP Test</title>
  </head>
  <body>
    <?php

      $enroll = htmlspecialchars($_GET["enroll"]);
      $name = htmlspecialchars($_GET["name"]);
      $attend = htmlspecialchars($_GET["attend"]);

      $conn = mysqli_connect('localhost', 'root', 'ssuhrid', 'test');
      if(! $conn ) {
        die('Could not connect: ' . mysqli_error($conn));
      }
      $sql1 = 'UPDATE test SET attendance = attendance + ' .$attend. ' WHERE enroll = ' .$enroll. '';
      $sql2 = 'INSERT INTO test (enroll,name,attendance) VALUES (' .$enroll. ',"' .$name. '",' .$attend. ')';
//      $sql = 'UPDATE test SET attendance = attendance + 10 WHERE enroll = 15102031' ;

      mysqli_select_db($conn,'test');
      $retval1 = mysqli_query( $conn, $sql1 );
      $retval2 = mysqli_query( $conn, $sql2 );
      if(! $retval1 ) {
        die('Could not enter data: ' . mysqli_error($conn));
      }
      
      if (! retval2){
        die('Could not enter data: ' . mysqli_error($conn));
      }
      echo "Entered data successfully\n";
      mysqli_close($conn);
    ?>

  </body>
</html>
