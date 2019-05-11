
import java.io.*;
import javafx.application.Application;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.stage.Stage;

public class AddressBook extends Application {

    // Specify the size of five string fields in the record
    final static int NAME_SIZE = 32;
    final static int STREET_SIZE = 32;
    final static int CITY_SIZE = 20;
    final static int STATE_SIZE = 2;
    final static int ZIP_SIZE = 5;
    final static int RECORD_SIZE
            = (NAME_SIZE + STREET_SIZE + CITY_SIZE + STATE_SIZE + ZIP_SIZE);

    // Access address.dat using RandomAccessFile
    private RandomAccessFile raf;

    // Text fields
    private TextField tfName = new TextField();
    private TextField tfStreet = new TextField();
    private TextField tfCity = new TextField();
    private TextField tfState = new TextField();
    private TextField tfZip = new TextField();

    // Buttons
    private Button btAdd = new Button("Add");
    private Button btFirst = new Button("First");
    private Button btNext = new Button("Next");
    private Button btPrevious = new Button("Previous");
    private Button btLast = new Button("Last");
    private Button btUpdate = new Button("Update");

    public AddressBook() {
        // Open or create a random access file
        try {
            raf = new RandomAccessFile("address.dat", "rw");
        } catch (IOException ex) {
            ex.printStackTrace();
            System.exit(1);
        }
    }

    @Override
    public void start(Stage primaryStage) {
        tfState.setPrefColumnCount(2);
        tfZip.setPrefColumnCount(4);
        tfCity.setPrefColumnCount(12);

        // Pane p1 for holding labels Name, Street, and City
        GridPane p1 = new GridPane();
        p1.setAlignment(Pos.CENTER);
        p1.setHgap(5);
        p1.setVgap(5);
        p1.add(new Label("Name"), 0, 0);
        p1.add(new Label("Street"), 0, 1);
        p1.add(new Label("City"), 0, 2);
        p1.add(tfName, 1, 0);
        p1.add(tfStreet, 1, 1);

        HBox p2 = new HBox(5);
        p2.getChildren().addAll(tfCity, new Label("State"), tfState,
                new Label("Zip"), tfZip);
        p1.add(p2, 1, 2);

        // Add buttons to a pane
        HBox p3 = new HBox(5);
        p3.getChildren().addAll(btAdd, btFirst, btNext, btPrevious,
                btLast, btUpdate);
        p3.setAlignment(Pos.CENTER);

        // Add p1 and p3 to a border pane
        BorderPane borderPane = new BorderPane();
        borderPane.setCenter(p1);
        borderPane.setBottom(p3);

        // Create a scene and place it in the stage
        Scene scene = new Scene(borderPane, 400, 120);
        primaryStage.setTitle("Address Book"); // Set the stage title
        primaryStage.setScene(scene); // Place the scene in the stage
        primaryStage.show(); // Display the stage

        // Display the first record if exists
        try {
            if (raf.length() > 0) {
                readAddress(0);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        /**
         * The Add button writes a new record at the end of the file
         */
        btAdd.setOnAction(e -> {
            try {
                // Write a record at the END of the file
                writeAddress(raf.length());
            } catch (Exception ex) {

            }
        });

        /**
         * The First button reads the record at the beginning of the file
         */
        btFirst.setOnAction(e -> {
            try {
                if (raf.length() > 0) {
                    readAddress(0);
                }
            } catch (IOException ex) {
            }
        });

        /**
         * The Next button reads the record at the current file pointer.
         */
        btNext.setOnAction(e -> {
            try {
                long currentPosition = raf.getFilePointer();
                if (currentPosition < raf.length()) {
                    readAddress(currentPosition);
                }
            } catch (IOException ex) {

            }
        });

        /**
         * The Previous button moves BACK two records from the current file
         * pointer. (Remember that the current file pointer points to the
         * location AFTER the record which is displayed.)
         */
        btPrevious.setOnAction(e -> {
            try {
                long currentPosition = raf.getFilePointer();
                if (currentPosition - 2 * RECORD_SIZE > 0) // (each char is 2 bytes)
                {
                    readAddress(currentPosition - 2 * 2 * RECORD_SIZE);
                } else {
                    readAddress(0);
                }
            } catch (IOException ex) {

            }
        });

        /**
         * The Last button reads the last record in the file, so it must point
         * the file pointer to the BEGINNING of that record.
         */
        btLast.setOnAction(e -> {
            try {
                long lastPosition = raf.length();
                if (lastPosition > 0) {
                    readAddress(lastPosition - 2 * RECORD_SIZE);
                }
            } catch (IOException ex) {
            }
        });
        
        btUpdate.setOnAction(e -> {
            try {
                long currentPosition = raf.getFilePointer() - 182;
                System.out.printf("Current Position: %d\n", currentPosition);
                System.out.printf("File Length: %d\n", raf.length());
                if (currentPosition < raf.length()) {
                    writeAddress(currentPosition);
                }
                else {
                    System.out.println("Cannot update an empty location in memory...");
                }
            }
            catch (IOException ex) {
                
            }
        });

    } // (end start method)

    /**
     * Write a record at the specified location in the file
     */
    public void writeAddress(long position) {
        try {
            raf.seek(position);
            FixedLengthStringIO.writeFixedLengthString(
                    tfName.getText(), NAME_SIZE, raf);
            FixedLengthStringIO.writeFixedLengthString(
                    tfStreet.getText(), STREET_SIZE, raf);
            FixedLengthStringIO.writeFixedLengthString(
                    tfCity.getText(), CITY_SIZE, raf);
            FixedLengthStringIO.writeFixedLengthString(
                    tfState.getText(), STATE_SIZE, raf);
            FixedLengthStringIO.writeFixedLengthString(
                    tfZip.getText(), ZIP_SIZE, raf);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    /**
     * Read a record at the specified position
     */
    public void readAddress(long position) throws IOException {
        raf.seek(position);
        String name = FixedLengthStringIO.readFixedLengthString(
                NAME_SIZE, raf);
        String street = FixedLengthStringIO.readFixedLengthString(
                STREET_SIZE, raf);
        String city = FixedLengthStringIO.readFixedLengthString(
                CITY_SIZE, raf);
        String state = FixedLengthStringIO.readFixedLengthString(
                STATE_SIZE, raf);
        String zip = FixedLengthStringIO.readFixedLengthString(
                ZIP_SIZE, raf);

        tfName.setText(name);
        tfStreet.setText(street);
        tfCity.setText(city);
        tfState.setText(state);
        tfZip.setText(zip);
    }

    /**
     * The main method is only needed for the IDE with limited JavaFX support.
     * Not needed for running from the command line.
     */
    public static void main(String[] args) {
        launch(args);
    }
} // (end class AddressBook)

class FixedLengthStringIO {

    /**
     * Read fixed number of characters from a DataInput stream. (A DataInput
     * stream is an object of any class which implements the DataInput
     * interface.)
     */
    public static String readFixedLengthString(int size,
            DataInput in) throws IOException {
        // Declare an array of characters
        char[] chars = new char[size];

        // Read fixed number of characters to the array
        for (int i = 0; i < size; i++) {
            chars[i] = in.readChar();
        }

        return new String(chars);
    }

    /**
     * Write fixed number of characters to a DataOutput stream. (A DataOutput
     * stream is an object of any class which implements the DataOutput
     * interface.)
     */
    public static void writeFixedLengthString(String s, int size,
            DataOutput out) throws IOException {
        char[] chars = new char[size];

        // Fill in string with characters
        s.getChars(0, s.length(), chars, 0);

        // Fill in blank characters in the rest of the array
        for (int i = Math.min(s.length(), size); i < chars.length; i++) {
            chars[i] = ' ';
        }

        // Create and write a new string padded with blank characters
        out.writeChars(new String(chars));
    }
}

